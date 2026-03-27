CREATE OR ALTER PROC sp5 (@TopN INT)
AS
BEGIN
    -- Check if the ReceiptsMostCloseToAverage table exists and drop it if it does
    IF OBJECT_ID('ReceiptsMostCloseToAverage', 'U') IS NOT NULL
    BEGIN
        DELETE FROM ReceiptsMostCloseToAverage;
    END
    ELSE
    BEGIN
        -- Create a new table ReceiptsMostCloseToAverage with the same columns as Transaction_Table
        SELECT TOP 0 *
        INTO ReceiptsMostCloseToAverage
        FROM Transaction_Table;
    END;

    -- Call and execute sp4 and insert the results into ReceiptsMostCloseToAverage
    INSERT INTO ReceiptsMostCloseToAverage (Receipt_No, Total_Payment) -- Specify columns to match sp4's implicit output
    EXEC sp4 @TopN;

    -- Display the contents of the ReceiptsMostCloseToAverage table
    SELECT * FROM ReceiptsMostCloseToAverage;
END;
GO

-- Testing script
EXEC sp5 5; -- Example: Call sp5 with an input value of 5