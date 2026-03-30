CREATE OR ALTER FUNCTION dbo.udf9 (@WeekendDate DATE)
RETURNS TABLE
AS
RETURN
(
    -- Check if the input date is a weekend (Saturday or Sunday)
    IF (DATENAME(weekday, @WeekendDate) = 'Saturday' OR DATENAME(weekday, @WeekendDate) = 'Sunday')
    BEGIN
        -- Find the top three largest receipts paid by credit card on the input weekend date
        SELECT TOP 3
            i.InvoiceID AS ReceiptNumber, 
            i.InvoiceDate AS ReceiptDate,
            'Credit Card' AS PaymentMethod, 
            i.InvoiceTotal AS TotalAmount
        FROM Invoices i 
        WHERE i.InvoiceDate = @WeekendDate
          AND i.PaymentMethod = 'Credit Card' 
        ORDER BY i.InvoiceTotal DESC;
    END
    ELSE
    BEGIN
        -- Return an empty table if the input date is not a weekend
        SELECT TOP 0
            NULL AS ReceiptNumber,
            NULL AS ReceiptDate,
            NULL AS PaymentMethod,
            NULL AS TotalAmount
        WHERE 1 = 0;
    END
);
Testing Script:
-- Test case 1: Input is a Saturday with potential receipts
SELECT 'Test Case 1: Saturday' AS TestDescription;
SELECT * FROM dbo.udf9('2024-11-09');
GO

-- Test case 2: Input is a Sunday with potential receipts
SELECT 'Test Case 2: Sunday' AS TestDescription;
SELECT * FROM dbo.udf9('2024-11-10');
GO
