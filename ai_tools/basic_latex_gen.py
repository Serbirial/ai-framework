def generate_latex_from_json(params: dict) -> dict:
    """
    Generate LaTeX code from structured JSON data.

    Expected JSON schema (example):
    {
      "type": "document",
      "content": [
        {"type": "section", "title": "Introduction"},
        {"type": "text", "text": "This is a sample paragraph."},
        {"type": "equation", "latex": "\\frac{a}{b} = c"},
        {"type": "table",
         "columns": ["A", "B"],
         "rows": [
            ["1", "2"],
            ["3", "4"]
         ]
        },
        {"type": "list", "ordered": False, "items": ["Apple", "Banana", "Cherry"]}
      ]
    }

    Supported types: document, section, text, equation, table, list

    Returns: {"latex_code": "..."} or {"error": "..."}
    """
    def render_element(el):
        t = el.get("type")
        if t == "document":
            content = el.get("content", [])
            return "\n\n".join(render_element(c) for c in content)
        elif t == "section":
            title = el.get("title", "Section")
            return f"\\section{{{title}}}"
        elif t == "text":
            return el.get("text", "")
        elif t == "equation":
            return f"\\[\n{el.get('latex','')}\n\\]"
        elif t == "table":
            cols = el.get("columns", [])
            rows = el.get("rows", [])
            col_format = "|".join(["c"] * len(cols))
            latex = "\\begin{tabular}{" + col_format + "}\n\\hline\n"
            latex += " & ".join(cols) + " \\\\\n\\hline\n"
            for row in rows:
                latex += " & ".join(row) + " \\\\\n"
            latex += "\\hline\n\\end{tabular}"
            return latex
        elif t == "list":
            items = el.get("items", [])
            ordered = el.get("ordered", False)
            env = "enumerate" if ordered else "itemize"
            latex = f"\\begin{{{env}}}\n"
            for item in items:
                latex += f"  \\item {item}\n"
            latex += f"\\end{{{env}}}"
            return latex
        else:
            return f"% Unknown element type: {t}"

    try:
        latex_code = render_element(params)
        return {"latex_code": latex_code}
    except Exception as e:
        return {"error": f"Failed to generate LaTeX: {str(e)}"}

EXPORT = {
    "generate_latex": {
        "help": (
            "Generate LaTeX code from structured JSON data describing the document.\n"
            "Supports: document, section, text, equation, table, list."
        ),
        "callable": generate_latex_from_json,
        "params": {
            "type": "document | section | text | equation | table | list",
            "content": "list of child elements (for document)",
            "title": "section title",
            "text": "text content",
            "latex": "raw latex string (for equation)",
            "columns": "list of column headers (for table)",
            "rows": "list of rows (for table)",
            "items": "list of items (for list)",
            "ordered": "bool, ordered or unordered list"
        }
    }
}
