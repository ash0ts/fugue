def invoice_total(subtotal: float, tax_rate: float, discount: float) -> float:
    """Return the final invoice amount rounded to cents."""

    return round(subtotal + subtotal * tax_rate + discount, 2)
