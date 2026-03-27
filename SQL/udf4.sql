CREATE OR ALTER FUNCTION udf4 (@ItemID INT)
RETURNS TABLE
AS
RETURN
(
    SELECT 
        t.Receipt_No,
        t.Date_of_Receipt,
        i.Item_Description,
        i.Unit_Price
    FROM Item_Table i
    JOIN Transaction_Table t ON i.Receipt_No = t.Receipt_No
    WHERE i.Item_No = @ItemID  -- Assuming you have a primary key column Item_No
);
GO

-- TESTING SCRIPT
SELECT * FROM dbo.udf4(1);  -- Replace 1 with a real Item_No
