-- Create the trigger trg9
CREATE OR ALTER TRIGGER trg9
ON dbo.Transaction_Table -- The trigger is on your Transaction_Table
AFTER INSERT, UPDATE -- It fires after Insert or Update operations 
AS
BEGIN
    -- Declare a variable to hold the daily total payment
    DECLARE @DailyTotal MONEY;

    -- Calculate the total payment for the day(s) affected by this operation.
    -- We sum the Total_Payment from the Transaction_Table for all rows
    -- whose Date_of_Receipt matches any date in the Inserted pseudo-table 
    -- This sum correctly includes the rows just inserted or updated within the current transaction.
    SELECT @DailyTotal = SUM(TT.Total_Payment)
    FROM dbo.Transaction_Table AS TT
    INNER JOIN (SELECT DISTINCT Date_of_Receipt FROM Inserted) AS I
        ON TT.Date_of_Receipt = I.Date_of_Receipt; -- Join on the date column 

    -- Check if the calculated daily total exceeds the limit of $1000
    IF @DailyTotal > 1000.00
    BEGIN
        -- If the limit is exceeded, raise a custom error and roll back the operation.
        -- THROW is used to raise an error and transfer execution to a CATCH block if one exists.
        -- State 1 is commonly used to indicate an error originating from a trigger or stored procedure
        -- and typically causes the statement that fired the trigger to be rolled back (based on common SQL Server error handling behavior).
        THROW 50003, 'Operation blocked: Daily total payment exceeds $1000.', 1; -- Use a unique error number
    END;

    -- If the daily total is within the limit, the trigger finishes without raising an error,
    -- allowing the original INSERT/UPDATE operation to proceed (within its transaction).

END;
GO

BEGIN TRAN; -- Start a transaction to isolate the test 

BEGIN TRY
    PRINT '--- Starting Test Case 1: Successful Operation (Insert 2 rows < $1000) ---';

    -- Use a date that likely has no existing transactions or very few,
    -- or use a date where existing transactions + these amounts < $1000.
    -- For isolation, let's assume we start fresh on '2023-10-29' within this transaction.

    -- Insert 2 sample rows into Transaction_Table with total payment significantly less than $1000
    INSERT INTO dbo.Transaction_Table (Receipt_No, Member_Name, sales_associate, Date_of_Receipt, Time_of_Receipt, Total_Payment, Payment_Method, Total_Tax, Street_Number_PK)
    VALUES
        (201, 'Customer I', 'Associate 3', '2023-10-29', '09:00:00', 400.00, 'Debit Card', 40.00, 1),
        (202, 'Customer J', 'Associate 4', '2023-10-29', '09:30:00', 350.00, 'Cash', 35.00, 1);
    -- Total for 2023-10-29 is now 400.00 + 350.00 = 750.00, which is <= 1000.00

    PRINT CAST(@@ROWCOUNT AS VARCHAR) + ' row(s) inserted into Transaction_Table.';

    -- Verify that the rows were successfully inserted
    PRINT 'Checking Transaction_Table for inserted rows...';
    SELECT Receipt_No, Member_Name, Date_of_Receipt, Total_Payment
    FROM dbo.Transaction_Table
    WHERE Date_of_Receipt = '2023-10-29' AND Receipt_No IN (201, 202); -- Should return 2 rows

    PRINT 'Test Case 1 successful: Rows inserted as expected.';

END TRY
BEGIN CATCH
    -- If an error occurs, print error details. This block should NOT be reached in Test Case 1.
    PRINT 'Test Case 1 failed due to an unexpected error:';
    PRINT 'Error Number: ' + CAST(ERROR_NUMBER() AS VARCHAR); 
    PRINT 'Error Message: ' + ERROR_MESSAGE(); 
    -- THROW; -- Re-throw the error if needed
END CATCH;

-- Rollback the transaction to undo all changes made during the test 
-- This removes the inserted rows.
IF @@TRANCOUNT > 0 -- Check if a transaction is active 
    ROLLBACK TRAN;

PRINT '--- Test Case 1 finished (transaction rolled back) ---';
GO

--Test Case two
BEGIN TRAN; -- Start a transaction to isolate the test [12, 14]

BEGIN TRY
    PRINT '--- Starting Test Case 2: Operation Blocked (Attempting to insert rows totaling > $1000) ---';

    -- Use the same date as before ('2023-10-29') or a new date.
    -- We will attempt to insert rows that by themselves exceed $1000,
    -- assuming no prior transactions for this date *within this transaction*.

    -- Attempt to insert 2 sample rows with total payment greater than $1000
    -- This should cause the trigger to fire and block the operation.
    INSERT INTO dbo.Transaction_Table (Receipt_No, Member_Name, sales_associate, Date_of_Receipt, Time_of_Receipt, Total_Payment, Payment_Method, Total_Tax, Street_Number_PK)
    VALUES
        (203, 'Customer K', 'Associate 5', '2023-10-29', '10:00:00', 600.00, 'Credit Card', 60.00, 1),
        (204, 'Customer L', 'Associate 6', '2023-10-29', '10:30:00', 500.00, 'Cash', 50.00, 1);
    -- Total for this INSERT would be 600.00 + 500.00 = 1100.00, which is > 1000.00

    -- If execution reaches here, the THROW statement in the trigger did NOT fire as expected.
    PRINT 'Error: Insert statement did NOT fail as expected!';

END TRY
BEGIN CATCH
    -- Catch block for handling the expected error from the trigger 
    -- Check if the caught error is the one thrown by trg9 (Error Number 50003, State 1)
    IF ERROR_NUMBER() = 50003 AND ERROR_STATE() = 1
    BEGIN
        PRINT 'Successfully caught expected error from trg9:';
        PRINT 'Error Number: ' + CAST(ERROR_NUMBER() AS VARCHAR); -- Display error number
        PRINT 'Error State: ' + CAST(ERROR_STATE() AS VARCHAR);
        PRINT 'Error Message: ' + ERROR_MESSAGE(); -- Display error message
        -- The THROW statement with a state > 0 in the trigger should automatically
        -- roll back the INSERT statement that caused it (based on common SQL Server behavior).
    END
    ELSE
    BEGIN
        -- Caught a different, unexpected error
        PRINT 'Caught an unexpected error:';
        PRINT 'Error Number: ' + CAST(ERROR_NUMBER() AS VARCHAR);
        PRINT 'Error Message: ' + ERROR_MESSAGE(); 
        -- THROW; -- Re-throw the unexpected error if necessary
    END
END CATCH;

-- Verify that no rows were inserted into Transaction_Table (due to the expected THROW and rollback)
PRINT 'Checking Transaction_Table for inserted rows...';
-- Use a WHERE clause that matches the attempted test data
SELECT Receipt_No, Member_Name, Date_of_Receipt, Total_Payment
FROM dbo.Transaction_Table
WHERE Date_of_Receipt = '2023-10-29' AND Receipt_No IN (203, 204); -- Should return 0 rows

-- Rollback the transaction to undo everything, just in case.
-- The error caught by TRY...CATCH usually handles the statement rollback,
-- but an explicit ROLLBACK TRAN ensures the entire test transaction is cleaned up.
IF @@TRANCOUNT > 0 
    ROLLBACK TRAN;

PRINT '--- Test Case 2 finished (transaction rolled back) ---';
GO