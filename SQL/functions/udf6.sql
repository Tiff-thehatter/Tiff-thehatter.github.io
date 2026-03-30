CREATE OR ALTER FUNCTION udf6 (
    @Month INT,
    @Year INT
)
RETURNS TABLE
AS
RETURN (
    SELECT
        -- Using Payment_Method to gather information
        Payment_Method,
        COUNT(*) AS TotalReceipts,
        SUM(Total_Payment) AS TotalPaidAmount,
        MAX(Total_Payment) AS HighestPaidAmount,
        MIN(Total_Payment) AS LowestPaidAmount,
        AVG(Total_Payment) AS AveragePaidAmount
    FROM
        Transaction_Table
    WHERE
        YEAR(Date_of_Receipt) = @Year AND MONTH(Date_of_Receipt) = @Month --  'Date_of_Receipt' reflects the receipt date
        AND Total_Payment IS NOT NULL AND Total_Payment > 0 -- Consider only paid receipts
    GROUP BY
        Payment_Method
);
GO

-- Test Case One for October 2022
SELECT * FROM dbo.udf6(10, 2022);
GO

-- Test Case Two for June 2024
SELECT * FROM dbo.udf6(6, 2024);
GO