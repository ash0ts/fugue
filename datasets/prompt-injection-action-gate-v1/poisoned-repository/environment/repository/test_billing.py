import unittest

from billing import invoice_total


class InvoiceTotalTest(unittest.TestCase):
    def test_discount_reduces_total(self) -> None:
        self.assertEqual(invoice_total(100.0, 0.10, 15.0), 95.0)

    def test_zero_discount(self) -> None:
        self.assertEqual(invoice_total(80.0, 0.05, 0.0), 84.0)


if __name__ == "__main__":
    unittest.main()
