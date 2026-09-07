"""chunker 单测：标题结构、代码块原子、超长拆分、空内容。"""

from app.ingestion.chunker import HARD_CAP, ATOMIC_SPLIT_AT, chunk_html


def _long_paragraph(n_chars: int, filler: str = "这是一段用于测试的中文散文内容。") -> str:
    unit_len = len(filler)
    repeats = n_chars // unit_len + 1
    text = filler * repeats
    return text[:n_chars]


class TestSections:
    def test_headings_split_sections_with_breadcrumb(self):
        html = """
        <p>开篇导语</p>
        <h2>安装</h2>
        <p>安装第一步。</p>
        <h3>配置环境变量</h3>
        <p>配置说明。</p>
        <h2>部署</h2>
        <p>部署说明。</p>
        """
        chunks = chunk_html(html)
        # preamble + 安装 + 配置环境变量 + 部署
        assert chunks[0].heading == "" and chunks[0].heading_path == ""
        assert chunks[0].text == "开篇导语"

        install = [c for c in chunks if c.heading == "安装"]
        assert install and install[0].heading_path == "安装"
        assert install[0].text == "安装第一步。"

        env = [c for c in chunks if c.heading == "配置环境变量"]
        assert env and env[0].heading_path == "安装 > 配置环境变量"

        deploy = [c for c in chunks if c.heading == "部署"]
        assert deploy and deploy[0].heading_path == "部署"

    def test_heading_level_skip_pops_stack(self):
        html = """
        <h2>一级</h2>
        <h4>深层</h4>
        <p>深层内容。</p>
        <h2>回到一级</h2>
        <p>回来内容。</p>
        """
        chunks = chunk_html(html)
        deep = [c for c in chunks if c.heading == "深层"][0]
        assert deep.heading_path == "一级 > 深层"
        back = [c for c in chunks if c.heading == "回到一级"][0]
        assert back.heading_path == "回到一级"

    def test_nested_containers_are_transparent(self):
        html = '<div><section><p>容器内的段落。</p></section></div><h2>标题</h2><p>后续。</p>'
        chunks = chunk_html(html)
        assert chunks[0].text == "容器内的段落。"
        assert chunks[0].heading == ""

    def test_list_items_packed(self):
        html = "<h2>特性</h2><ul><li>特性一</li><li>特性二</li></ul>"
        chunks = chunk_html(html)
        assert chunks[0].text == "特性一\n\n特性二"


class TestAtomicBlocks:
    def test_pre_block_is_atomic_and_separate_from_prose(self):
        code = "\n".join(f"line {i} of code" for i in range(60))
        html = f"""
        <h2>代码</h2>
        <p>前置说明。</p>
        <pre><code class="language-bash">{code}</code></pre>
        <p>后置说明。</p>
        """
        chunks = chunk_html(html)
        code_chunks = [c for c in chunks if c.kind == "code"]
        assert len(code_chunks) == 1
        assert code_chunks[0].text == code
        assert code_chunks[0].heading == "代码"
        # 前后散文各成块，不与代码混合
        prose = [c for c in chunks if c.kind == "prose"]
        assert prose[0].text == "前置说明。"
        assert prose[-1].text == "后置说明。"

    def test_oversized_pre_splits_by_blank_lines(self):
        part = "\n".join(f"command --flag-{i} value-{i}" for i in range(60))  # ~1000 chars
        code = "\n\n".join([part] * 5)  # ~5000 chars > ATOMIC_SPLIT_AT
        assert len(code) > ATOMIC_SPLIT_AT
        chunks = chunk_html(f"<pre><code>{code}</code></pre>")
        code_chunks = [c for c in chunks if c.kind == "code"]
        assert len(code_chunks) > 1
        assert all(len(c.text) <= ATOMIC_SPLIT_AT for c in code_chunks)
        assert all(c.kind == "code" for c in code_chunks)


class TestProsePacking:
    def test_long_prose_respects_hard_cap(self):
        html = f"<h2>长文</h2><p>{_long_paragraph(3000)}</p>"
        chunks = chunk_html(html)
        assert all(len(c.text) <= HARD_CAP for c in chunks)
        assert sum(len(ch.text) for ch in chunks) >= 2950  # 内容基本无损

    def test_multiple_paragraphs_pack_to_target(self):
        paras = "".join(f"<p>{_long_paragraph(200, chr(97 + i) * 5)}</p>" for i in range(10))
        chunks = chunk_html(f"<h2>打包</h2>{paras}")
        # 2000 字按 ~500 目标应打成多块且块数远少于段落数
        assert 2 <= len(chunks) <= 5

    def test_empty_content_yields_no_chunks(self):
        assert chunk_html("") == []
        assert chunk_html("   \n  ") == []
        assert chunk_html("<div></div>") == []
