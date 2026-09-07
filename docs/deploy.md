# blog-agent 生产部署手册

目标环境：腾讯云服务器（博客与 MySQL 所在的那台，下称"服务器"），Docker 方案。

## 架构（部署后）

```
访客浏览器
  │  https://博客前台域名
  ▼
nginx（服务器上已有的）
  ├─ 静态文件 / Laravel API（原有配置不动）
  └─ location /agent-api/ ──反代──→ 127.0.0.1:8000
                                        │
                              docker compose: agent + qdrant
                                        ▲
Laravel Observer → 队列 Job → POST 127.0.0.1:8000/api/admin/sync-article
```

同域路径反代的好处：**不需要新域名/证书，也不需要 CORS**。

## 一、准备（本地）

1. `.env.production` 填真实值：
   ```bash
   # 生成强 token（与 Laravel 侧共用同一个值）
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
   - `CHAT_API_KEY` / `EMBEDDING_API_KEY`：真实 key（可与开发同 key）
   - `ADMIN_SYNC_TOKEN`：上面的强随机串
   - `BLOG_API_BASE=https://blog-admin.lqy-comic.com/api`
   - `CHAT_CORS_ORIGINS`：留空（走同域反代）
2. 代码入库：`git init && git add . && git commit`（.env.production 已被 gitignore，**不要提交**，单独传服务器）

## 二、服务器上部署 agent + Qdrant

```bash
# 1. 代码上服务器（git clone 或 rsync，.env.production 单独 scp）
rsync -av --exclude data --exclude .venv ./ user@服务器:/opt/blog-agent/
scp .env.production user@服务器:/opt/blog-agent/.env.production

# 2. 启动（首次会拉镜像 + 构建几分钟）
cd /opt/blog-agent && docker compose up -d --build

# 3. 首次全量同步文章（45 篇约 1 分钟）
docker compose exec agent .venv/bin/python -m scripts.sync_articles

# 4. 验证
curl http://127.0.0.1:8000/api/health
# 期望 {"status":"ok","env":"production","qdrant":true,"points":700 左右}
```

## 三、nginx 反代（SSE 关键配置）

在博客前台站点（blog-vue 静态文件所在）的 server 块里加：

```nginx
location /agent-api/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

    # SSE 流式必需：关闭缓冲，否则回答会攒成一坨最后一次性吐出
    proxy_buffering off;
    proxy_cache off;
    # LLM 生成长回答可能较慢，读超时放宽
    proxy_read_timeout 300s;
}
```

`nginx -t && nginx -s reload` 后验证：`curl -N https://前台域名/agent-api/api/health`。

## 四、blog-vue 前端

```ini
# .env.production
VITE_CHAT_API_BASE_URL=/agent-api   # 同域相对路径
```

然后正常构建发布（`npm run build`），npm 依赖装好（markdown-it / dompurify 已在 package.json）。

## 五、Laravel 侧（服务器上的 blog-admin-v2）

1. 拉取包含以下内容的新代码：`SyncArticleToBlogAgent` Job、`ArticleObserver` 改动、`config/blog_agent.php`
2. 服务器 `.env` 追加：
   ```ini
   BLOG_AGENT_ENABLED=true
   BLOG_AGENT_URL=http://127.0.0.1:8000     # 同机部署直连
   BLOG_AGENT_SYNC_TOKEN=<与 agent .env.production 一致的强随机串>
   ```
3. `php artisan config:clear`
4. **队列 worker 加监听 blog-agent 队列**（改 supervisor / systemd 里 worker 的命令后重启）：
   ```bash
   php artisan queue:work --queue=default,blog-agent --tries=3
   ```

## 六、验收清单

- [ ] `curl https://前台域名/agent-api/api/health` → `env=production`，points 数正常
- [ ] 前台打开聊天气泡提问 → 流式打字机输出、引用可跳转、可追问
- [ ] 后台编辑任意文章保存 → `docker compose logs -f agent` 几秒内出现 `single sync: indexed ...`
- [ ] 后台下架文章 → agent 日志出现 `not found upstream, deleted points`
- [ ] 错 token 调 `/api/admin/sync-article` 返回 401

## 运维备忘

| 事项 | 命令 |
|---|---|
| 看日志 | `docker compose logs -f agent` |
| 重启 | `docker compose restart agent` |
| 手动全量同步 | `docker compose exec agent .venv/bin/python -m scripts.sync_articles --full` |
| 备份 | `docker run --rm -v <项目名>_agent_data:/data -v $(pwd):/backup alpine tar czf /backup/agent-data.tgz /data`（会话 sqlite）；知识库可随时重跑全量同步重建 |
| 升级 | `git pull && docker compose up -d --build` |
| 换 embedding 模型 | 改 `.env.production` → `docker compose down` → 删 qdrant 卷 → `up -d` → 全量同步 |

## 排错

- **容器起不来，日志报 `Missing credentials`**：`.env.production` 的 key 为空——服务启动会构建
  embedding/LLM 客户端，key 缺失直接快速失败（不会带病运行）。检查 `env_file` 是否被加载。
- **`sqlite3.OperationalError: unable to open database file`**：data 目录容器内无写权限。
  已用 named volume（`agent_data`）解决——首次挂载自动继承镜像内 app 用户属主。
  若改回 bind mount（`./data`），宿主机需 `mkdir -p data && chown -R 1000:1000 data`（镜像固定 UID 1000）。
- **宿主机直接跑脚本报 `Name or service not known`**：没带 `APP_ENV`，读了 `.env.development`
  里的本地开发域名（www.blog.test 仅存在于开发机）。宿主机跑须 `APP_ENV=production uv run ...`；
  推荐统一走容器（compose 已注入正确的 env_file）。
- **前端回答不是流式（攒一坨一次出现）**：nginx 没关缓冲，检查 `proxy_buffering off`。
- **限流把所有用户当成一个人**：直连 uvicorn 时 `request.client.host` 是反代 IP——镜像 CMD 已带
  `--proxy-headers --forwarded-allow-ips=127.0.0.1`，nginx 需传 `X-Forwarded-For`（上面配置已含）。
