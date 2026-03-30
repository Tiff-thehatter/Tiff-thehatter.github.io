CREATE OR ALTER FUNCTION dbo.udf8 (@Input INT)
RETURNS TABLE
AS
RETURN
(
    SELECT TOP 1
        DATENAME(dw, InvoiceDate) AS Weekday,
        AVG(InvoiceTotal) AS AverageExpense
    FROM
        Invoices
    GROUP BY
        DATENAME(dw, InvoiceDate)
    ORDER BY
        CASE @Input
            WHEN 0 THEN AVG(InvoiceTotal) ASC
            WHEN 1 THEN AVG(InvoiceTotal) DESC
        END
);
GO

-- Testing script
SELECT 'Lowest Average Expense Weekday' AS AnalysisType;
SELECT * FROM dbo.udf8(0);

SELECT 'Highest Average Expense Weekday' AS AnalysisType;
SELECT * FROM dbo.udf8(1);