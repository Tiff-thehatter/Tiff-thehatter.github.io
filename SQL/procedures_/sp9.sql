CREATE OR ALTER PROCEDURE sp9
    @TotalWeekendExpense MONEY OUTPUT
AS
BEGIN
    SET NOCOUNT ON;
    SET @TotalWeekendExpense = 0;

    -- Create a temporary table to store distinct weekend dates
    DROP TABLE IF EXISTS #WeekendDates;
    CREATE TABLE #WeekendDates (
        WeekendDay DATE PRIMARY KEY
    );

    -- Populate the temporary table with distinct weekend dates from Invoices
    INSERT INTO #WeekendDates (WeekendDay)
    SELECT DISTINCT InvoiceDate
    FROM Invoices
    WHERE DATENAME(weekday, InvoiceDate) IN ('Saturday', 'Sunday');

    -- Declare a cursor to iterate through the distinct weekend dates
    DECLARE weekend_cursor CURSOR FOR
    SELECT WeekendDay
    FROM #WeekendDates
    ORDER BY WeekendDay;

    DECLARE @CurrentWeekendDay DATE;

    OPEN weekend_cursor;

    FETCH NEXT FROM weekend_cursor INTO @CurrentWeekendDay;

    -- Iterate through each weekend day
    WHILE @@FETCH_STATUS = 0
    BEGIN
        PRINT '--- Weekend Day: ' + CONVERT(VARCHAR, @CurrentWeekendDay, 101) + ' ---';

        -- Call udf9 to get the top three largest credit card receipts for the current weekend day
        SELECT *
        INTO #TopReceipts
        FROM dbo.udf9(@CurrentWeekendDay);

        -- Display the top receipts for the current weekend day
        SELECT *
        FROM #TopReceipts;

        -- Calculate the sum of the total amounts for the top receipts
        DECLARE @CurrentWeekendExpense MONEY;
        SELECT @CurrentWeekendExpense = ISNULL(SUM(TotalAmount), 0)
        FROM #TopReceipts;

        -- Add the current weekend's expense to the total
        SET @TotalWeekendExpense = @TotalWeekendExpense + @CurrentWeekendExpense;

        DROP TABLE #TopReceipts;

        FETCH NEXT FROM weekend_cursor INTO @CurrentWeekendDay;
    END;

    CLOSE weekend_cursor;
    DEALLOCATE weekend_cursor;

    -- Drop the temporary table
    DROP TABLE IF EXISTS #WeekendDates;
END;

-- Testing Script for sp9
DECLARE @WeekendExpenseResult MONEY;

EXEC sp9 @WeekendExpenseResult OUTPUT;

SELECT 'Total Expense of Top 3 Receipts for All Weekend Days: $' + CONVERT(VARCHAR, @WeekendExpenseResult, 1);
GO