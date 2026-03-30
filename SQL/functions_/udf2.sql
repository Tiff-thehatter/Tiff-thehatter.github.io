USE [EERDBS-3];
GO

------------------------------------------------------------
-- 1) CREATE OR ALTER FUNCTION: udf2
--    Returns a table (Month, Year, Average Expense)
------------------------------------------------------------
CREATE OR ALTER FUNCTION dbo.udf2()
RETURNS @MonthlyAverages TABLE 
(
    [Month]           NVARCHAR(20),
    [Year]            INT,
    [Average Expense] MONEY
)
AS
BEGIN
    INSERT INTO @MonthlyAverages
    SELECT 
       DATENAME(MONTH, TRY_CAST(Date_of_Receipt AS DATE)) AS [Month],
       YEAR(TRY_CAST(Date_of_Receipt AS DATE))            AS [Year],
       AVG(Total_Payment)                                 AS [Average Expense]
    FROM dbo.Transaction_Table
    -- Exclude rows where the date cannot be parsed
    WHERE TRY_CAST(Date_of_Receipt AS DATE) IS NOT NULL
    GROUP BY 
       DATENAME(MONTH, TRY_CAST(Date_of_Receipt AS DATE)),
       YEAR(TRY_CAST(Date_of_Receipt AS DATE));

    RETURN;
END;
GO

------------------------------------------------------------
-- 2) TESTING SECTION
--    Example query to test udf2
------------------------------------------------------------
SELECT *
FROM dbo.udf2();
GO
