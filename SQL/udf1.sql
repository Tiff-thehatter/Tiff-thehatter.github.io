USE [EERDBS-3];
GO

----------------------------------------------------
-- 1) CREATE OR ALTER FUNCTION: udf1
--    Returns the total expense of all receipts
----------------------------------------------------
CREATE OR ALTER FUNCTION dbo.udf1()
RETURNS MONEY
AS
BEGIN
    DECLARE @TotalExpense MONEY;

    SELECT @TotalExpense = SUM(Total_Payment)
    FROM dbo.Transaction_Table;

    RETURN ISNULL(@TotalExpense, 0);
END;
GO

----------------------------------------------------
-- 2) TESTING SECTION
--    Example query to test udf1
----------------------------------------------------
SELECT dbo.udf1() AS [Total_Expense];
GO
