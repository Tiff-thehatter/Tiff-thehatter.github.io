USE [EERDBS-3]
GO
CREATE OR ALTER FUNCTION dbo.udf7 (
    @MinUnitPrice money,
    @MaxUnitPrice money
)
RETURNS TABLE
AS
RETURN (
    SELECT
        it.Receipt_No AS recieptid,
        it.Item_Description AS itemdescription,
        it.Unit_Price AS price,
        it.I_D AS ItemID
    FROM
        Item_Table it
    WHERE
        it.Unit_Price BETWEEN @MinUnitPrice AND @MaxUnitPrice
);
GO
--Test Case One:
PRINT 'Testing dbo.udf7 ranging from $20-$90'

SELECT * FROM dbo.udf7(200.00,300.00)
GO