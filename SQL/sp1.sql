USE [EERDBS-3];
GO

---------------------------------------------------------
-- 1) CREATE OR ALTER PROCEDURE: sp1
--    Finds and displays the largest five receipts in
--    terms of each receipt's total amount. Uses a cursor
--    to display:
--         rank, receipt ID, total amount, difference
---------------------------------------------------------
CREATE OR ALTER PROCEDURE dbo.sp1
AS
BEGIN
    SET NOCOUNT ON;

    -------------------------------------------------------------------
    -- 1A. Get the top 5 receipts in descending order by total amount
    --     Store them in a temporary table with an additional
    --     "RowNum" column to represent each row’s rank.
    -------------------------------------------------------------------
    CREATE TABLE #TopFiveReceipts
    (
        RowNum       INT,         -- rank
        ReceiptNo    TINYINT,     -- or match your actual data type for Receipt_No
        TotalAmount  MONEY
    );

    ;WITH CTE_Top AS
    (
        SELECT TOP (5)
               Receipt_No,
               Total_Payment
        FROM dbo.Transaction_Table
        ORDER BY Total_Payment DESC
    )
    INSERT INTO #TopFiveReceipts
    SELECT
        ROW_NUMBER() OVER (ORDER BY Total_Payment DESC) AS RowNum,
        Receipt_No,
        Total_Payment
    FROM CTE_Top
    ORDER BY Total_Payment DESC;

    -------------------------------------------------------------------
    -- 1B. Declare a cursor to walk through these top-5 rows
    -------------------------------------------------------------------
    DECLARE 
        @Rank          INT,
        @ReceiptNo     TINYINT,
        @CurrentAmount MONEY,
        @PrevAmount    MONEY = 0,   -- store the previous (larger) total
        @Difference    MONEY;

    DECLARE topReceiptsCursor CURSOR FOR
        SELECT RowNum, ReceiptNo, TotalAmount
        FROM #TopFiveReceipts
        ORDER BY RowNum;  -- ensures we process rank 1 first, then 2, etc.

    OPEN topReceiptsCursor;

    FETCH NEXT FROM topReceiptsCursor
    INTO @Rank, @ReceiptNo, @CurrentAmount;

    -------------------------------------------------------------------
    -- 1C. Loop through each row, compute the difference vs. predecessor
    -------------------------------------------------------------------
    WHILE @@FETCH_STATUS = 0
    BEGIN
        IF @Rank = 1
            SET @Difference = 0;  -- first item has no predecessor
        ELSE
            SET @Difference = @PrevAmount - @CurrentAmount;

        -------------------------------------------------------------------
        -- 1D. Display the row
        -------------------------------------------------------------------
        SELECT 
            @Rank            AS [rank],
            @ReceiptNo       AS [receipt ID],
            @CurrentAmount   AS [Total Amount],
            @Difference      AS [Difference];

        SET @PrevAmount = @CurrentAmount;  -- track last total for next loop

        FETCH NEXT FROM topReceiptsCursor
        INTO @Rank, @ReceiptNo, @CurrentAmount;
    END

    CLOSE topReceiptsCursor;
    DEALLOCATE topReceiptsCursor;

    -- Clean up
    DROP TABLE #TopFiveReceipts;
END;
GO

---------------------------------------------------------
-- 2) TESTING sp1
---------------------------------------------------------
EXEC dbo.sp1;
GO
