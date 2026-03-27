CREATE OR ALTER PROC sp4
    @TopN INT
AS
BEGIN
    DECLARE @totalExpense MONEY;
    DECLARE @totalDays INT;
    DECLARE @dailyAvg MONEY;

    -- Get total expense from udf1
    SELECT @totalExpense = dbo.udf1();

    -- Count distinct days with receipts
    SELECT @totalDays = COUNT(DISTINCT Date_of_Receipt)
    FROM Transaction_Table;

    SET @dailyAvg = @totalExpense / @totalDays;

    -- Select receipts closest to the daily average
    ;WITH ReceiptDiff AS (
        SELECT 
            Receipt_No,
            Total_Payment,
            ABS(Total_Payment - @dailyAvg) AS Diff
        FROM Transaction_Table
    )
    SELECT *
    FROM ReceiptDiff
    WHERE Diff IN (
        SELECT TOP (@TopN) Diff
        FROM ReceiptDiff
        ORDER BY Diff ASC
    )
    ORDER BY Diff ASC;
END;
GO

-- TESTING SCRIPT
EXEC sp4 @TopN = 3;
