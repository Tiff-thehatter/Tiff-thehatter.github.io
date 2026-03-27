USE [EERDBS-3];
GO

------------------------------------------------------------
-- 1) CREATE OR ALTER PROCEDURE: sp2
--    Output Parameters:
--        @ReceiptCount   INT
--        @AveragePayment MONEY
--    Functionality:
--      - Calculate number of receipts
--      - Calculate overall average total amount
--      - Display receipts >= 120% of that average
------------------------------------------------------------
CREATE OR ALTER PROCEDURE dbo.sp2
    @ReceiptCount   INT    OUTPUT,
    @AveragePayment MONEY  OUTPUT
AS
BEGIN
    SET NOCOUNT ON;

    ---------------------------------------------------------
    -- Step A: Calculate total receipt count and average amount
    ---------------------------------------------------------
    SELECT 
        @ReceiptCount   = COUNT(*),
        @AveragePayment = AVG(Total_Payment)
    FROM dbo.Transaction_Table;

    ---------------------------------------------------------
    -- Step B: Display receipts whose total is at least
    --         20% higher than overall average
    ---------------------------------------------------------
    SELECT 
        Receipt_No     AS [Receipt ID],
        Total_Payment  AS [Total Amount]
    FROM dbo.Transaction_Table
    WHERE Total_Payment >= 1.2 * @AveragePayment;

END;
GO

------------------------------------------------------------
-- 2) TESTING SECTION
-- Example usage:
--   - Declare local variables for output parameters
--   - Execute procedure
--   - View the resulting output parameter values
------------------------------------------------------------
DECLARE @count INT, @avg MONEY;

EXEC dbo.sp2 
    @ReceiptCount = @count OUTPUT,
    @AveragePayment = @avg OUTPUT;

SELECT 
    @count AS [Total Number of Receipts], 
    @avg   AS [Average of All Receipts];
GO
