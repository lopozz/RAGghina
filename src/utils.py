def create_retrieval_context_section(documents: list[str]) -> str:
    """
    Creates an XML-style <context> section from a list of document strings.
    with <document> tags.
    """
    if not documents:
        return ''
    doc_tags = "\n\n".join(
        f'<document">\n{doc.strip()}\n</document>'
        for i, doc in enumerate(documents)
    )
    return f"<context>\n{doc_tags}\n</context>"
