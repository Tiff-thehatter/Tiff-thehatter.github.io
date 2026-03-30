CREATE OR ALTER FUNCTION dbo.udf5 ()
RETURNS TABLE
AS
RETURN
(
    SELECT TOP 5
        InvoiceNumber AS ReceiptNumber,
        InvoiceTotal AS TotalAmount
    FROM
        Invoices
    ORDER BY
        InvoiceTotal DESC
);
GO

-- Testing script
SELECT * FROM dbo.udf5();