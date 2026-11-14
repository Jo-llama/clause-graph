from clause_graph.extract import (
    extract,
    find_definitions,
    find_references,
    resolve_reference,
    split_sections,
)

MINI = """SUPPLY AGREEMENT
This Supply Agreement is made between Acme Ltd ("Supplier") and Beta GmbH ("Buyer").

1. Definitions. "Losses" means all losses, damages, costs and expenses.

12. Limitation of Liability. The Supplier's total liability for Losses shall not exceed the fees paid.

15. Exclusions. Section 12 shall not apply to fraud or wilful misconduct.
"""


def by_section(graph):
    return {c.section: c.uid for c in graph.clauses}


def test_mini_contract_sections():
    assert [s.number for s in split_sections(MINI)] == ["0", "1", "12", "15"]


def test_mini_contract_graph():
    g = extract("mini", MINI)
    uid = by_section(g)

    assert (uid["0"], "Supplier") in g.defines
    assert (uid["0"], "Buyer") in g.defines
    assert (uid["1"], "Losses") in g.defines
    assert (uid["12"], "Losses") in g.uses

    refs_from_15 = {to for frm, to, _ in g.references if frm == uid["15"]}
    assert refs_from_15 == {uid["12"]}
    assert g.unresolved == []
    assert g.metrics["sections"] == 3
    assert g.metrics["unresolved_refs"] == 0


def test_uppercase_and_list_references():
    text = "subject to ARTICLE 5 and Sections 2.7, 8 and 10, and clauses 3 through 4.1"
    assert find_references(text) == ["5", "2.7", "8", "10", "3", "4.1"]


def test_year_and_large_integers_are_not_sections():
    text = "Preamble.\n1. Term. Starts in\n2019 Annual Report.\n45 Days notice applies.\n2. Fees. None."
    assert [s.number for s in split_sections(text)] == ["0", "1", "2"]


def test_repeated_numbers_keep_first_occurrence():
    text = "1. Scope. Text.\n2. Price. Table:\n1. Widgets 10\n3. Term. Text."
    assert [s.number for s in split_sections(text)] == ["1", "2", "3"]


def test_definition_forms():
    text = (
        'the products (collectively, "Products") and the seller (the "Seller"). '
        '"Territory" shall have the meaning set out below.'
    )
    assert find_definitions(text) == ["Products", "Seller", "Territory"]


def test_resolve_reference_falls_back_to_first_subsection():
    sections = ["0", "1", "12.1", "12.2"]
    assert resolve_reference("12", sections) == "12.1"
    assert resolve_reference("1", sections) == "1"
    assert resolve_reference("7", sections) is None


def test_no_self_references_and_hub_terms():
    text = (
        'This Agreement ("Agreement").\n'
        "1. Scope. This Agreement covers it. See Section 1.\n"
        "2. Term. This Agreement lasts.\n"
        "3. Law. This Agreement is governed.\n"
    )
    g = extract("hub", text)
    assert g.references == []
    assert g.hub_terms == ["Agreement"]
    assert all(name != "Agreement" for _, name in g.uses)
