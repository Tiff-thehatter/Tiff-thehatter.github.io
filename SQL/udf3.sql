CREATE OR ALTER FUNCTION udf3()
RETURNS TABLE
AS
RETURN
(
    SELECT TOP 1 
        Item_Description,
        Unit_Price
    FROM Item_Table
    ORDER BY Unit_Price DESC
);
GO

-- TESTING SCRIPT
SELECT * FROM dbo.udf3();
