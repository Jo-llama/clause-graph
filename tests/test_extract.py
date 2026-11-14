from clause_graph.extract import (
    extract,
    find_definitions,
    find_parent,
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


# ---------------------------------------------------------------- step 3b: splitting


def test_spaced_numbering_is_normalised():
    text = (
        "Preamble.\n5.    LICENSE FEES   5. 1.     Licence Fee: payable now.\n"
        "5 .2 Payment Terms: thirty days.\n23. 1   Without limiting the above, insurance."
    )
    assert [s.number for s in split_sections(text)] == ["0", "5", "5.1", "5.2", "23.1"]


def test_find_parent_walks_up_to_nearest_existing_ancestor():
    sections = {"9.0", "9.1", "12", "12.1", "12.1.1", "14.2.3"}
    assert find_parent("9.1", sections) == "9.0"  # article exists only as a "9.0" heading
    assert find_parent("12.1", sections) == "12"
    assert find_parent("12.1.1", sections) == "12.1"
    assert find_parent("14.2.3", sections) is None  # no ancestor extracted
    assert find_parent("12", sections) is None
    assert find_parent("9.0", sections | {"9"}) == "9"
    assert find_parent("H3", sections) is None


def test_subsection_links():
    text = (
        "Preamble.\n8.1 Indemnity. Each Party indemnifies.\n"
        "8.2 Cap. Liability shall not exceed the insurance in Section 9.0.\n"
        "9.0    INSURANCE\n9.1 Cover. Cargo insurance of $100,000.\n9.2 Proof. Certificates.\n"
    )
    g = extract("dyn", text)
    uid = by_section(g)
    assert set(g.subsections) == {(uid["9.1"], uid["9.0"]), (uid["9.2"], uid["9.0"])}
    assert g.metrics["subsection_links"] == 2


CAPS = """GLOBAL MASTER SUPPLY AGREEMENT

This Agreement is between Seller Inc ("Seller") and Buyer Inc ("Buyer").

PRODUCTS, QUANTITY, PRICE

Buyer shall purchase the Product.

ACME CORP CONFIDENTIAL
Page text continues here.

WRITTEN AGREEMENT If there is an executed contract, it prevails.

IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT OR CONSEQUENTIAL DAMAGES WHATSOEVER.
ACME CORP CONFIDENTIAL
More text.
ACME CORP CONFIDENTIAL
End.
"""


def test_caps_headings_become_synthetic_sections():
    secs = split_sections(CAPS)
    assert [s.number for s in secs] == ["0", "H1", "H2"]
    assert secs[0].text.startswith("GLOBAL MASTER SUPPLY AGREEMENT")  # title stays in preamble
    assert secs[1].text.startswith("PRODUCTS, QUANTITY, PRICE")
    assert secs[2].text.startswith("WRITTEN AGREEMENT If there")
    # repeated page header and the all-caps disclaimer do not split
    assert "ACME CORP CONFIDENTIAL" in secs[1].text and "IN NO EVENT" in secs[2].text


def test_caps_headings_can_be_disabled_and_do_not_count_as_numbered():
    assert [s.number for s in split_sections(CAPS, caps_headings=False)] == ["0"]
    g = extract("caps", CAPS)
    assert g.metrics["sections"] == 0  # numbering failure stays visible
    assert g.metrics["caps_sections"] == 2
    assert [c.position for c in g.clauses] == [0, 1, 2]


def test_size_metrics():
    g = extract("mini", MINI)
    assert g.metrics["preamble_chars"] == len(g.clauses[0].text)
    assert g.metrics["max_clause_chars"] == max(len(c.text) for c in g.clauses)
