CREATE OR ALTER TRIGGER trg8
ON dbo.Transaction_Table -- Assuming Transaction_Table is the correct table name
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    -- Declare variables
    DECLARE @InsertedCount INT = (SELECT COUNT(*) FROM Inserted);
    DECLARE @DeletedCount INT = (SELECT COUNT(*) FROM Deleted);
    DECLARE @AffectedRows INT = 0;
    DECLARE @OperationType CHAR(1);
    DECLARE @TodayLogCount INT = 0;
    DECLARE @MaxDailyLimit INT = 5; -- The daily limit set by the trigger

    -- Determine operation type and affected row count
    IF @InsertedCount > 0 AND @DeletedCount = 0
    BEGIN
        SET @AffectedRows = @InsertedCount;
        SET @OperationType = 'I';
    END
    ELSE IF @InsertedCount = 0 AND @DeletedCount > 0
    BEGIN
        SET @AffectedRows = @DeletedCount;
        SET @OperationType = 'D';
    END
    ELSE IF @InsertedCount > 0 AND @DeletedCount > 0
    BEGIN
        SET @AffectedRows = @InsertedCount; -- For UPDATE, Inserted count represents affected rows
        SET @OperationType = 'U';
    END
    ELSE
    BEGIN
        -- No rows affected (e.g., DELETE/UPDATE with no matching rows)
        RETURN;
    END

    -- Check if any rows were actually affected before logging/checking limit
    IF @AffectedRows > 0
    BEGIN
        -- Get the current total affected rows logged for today
        SELECT @TodayLogCount = COALESCE(SUM(ReceiptAffected), 0) -- Use ReceiptAffected column name from source [1]
        FROM ReceiptsLog
        WHERE LogDate = CAST(GETDATE() AS DATE);

        -- Check against the daily limit
        IF @TodayLogCount + @AffectedRows > @MaxDailyLimit
        BEGIN
            -- Operation exceeds limit, throw error and implicitly roll back statement
            THROW 50002, 'Operation blocked by trigger trg8: Exceeds daily limit of 5 affected receipts.', 1;
        END
        ELSE
        BEGIN
            -- Limit not exceeded, log the operation
            INSERT INTO ReceiptsLog (LogDate, LoginName, UserName, OperationType, ReceiptAffected) -- Correct column list
            VALUES (CAST(GETDATE() AS DATE), SYSTEM_USER, USER_NAME(), @OperationType, @AffectedRows); -- Provide values for all columns
        END
    END
END;
GO

-- Test Case 1: Successful Operation (Logging Occurs)
-- Assumes a valid Street_Number_PK exists in Shop_Table (e.g., 1)
-- Assumes Receipt_No values 101 and 102 are not already used in Transaction_Table and fit the tinyint type range (0-255) [3].
BEGIN TRAN; -- Start a transaction to isolate the test [9, 11, 12]

BEGIN TRY
    PRINT '--- Starting Test Case 1: Successful Operation (Insert 2 rows) ---';

    -- Insert 2 sample rows into Transaction_Table using actual column names [3]
    -- Provide values for all NOT NULL columns: Receipt_No (PK, tinyint) [3], Member_Name (nvarchar) [3], Date_of_Receipt (nvarchar) [3], Time_of_Receipt (time) [3], Total_Payment (money) [3], Total_Tax (money) [3]
    -- Provide values for nullable columns: sales_associate (nvarchar) [3], Payment_Method (nvarchar) [3], Street_Number_PK (smallint, FK to Shop_Table) [3, 4]
    INSERT INTO dbo.Transaction_Table (Receipt_No, Member_Name, sales_associate, Date_of_Receipt, Time_of_Receipt, Total_Payment, Payment_Method, Total_Tax, Street_Number_PK)
    VALUES
        (211, 'Customer A', 'Associate 1', '2023-10-28', '10:00:00', 50.00, 'Cash', 5.00, 1), -- Example Street_Number_PK = 1
        (212, 'Customer B', 'Associate 2', '2023-10-28', '11:30:00', 75.50, 'Credit Card', 7.55, 1);

    PRINT CAST(@@ROWCOUNT AS VARCHAR) + ' row(s) inserted into Transaction_Table.';

    -- Verify that a log entry was created by trg8
    -- The log entry should reflect the operation type 'I' and affected count 2
    PRINT 'Checking ReceiptsLog for today''s entry...';
    SELECT LogDate, LoginName, UserName, OperationType, ReceiptAffected
    FROM ReceiptsLog
    WHERE LogDate = CAST(GETDATE() AS DATE) AND OperationType = 'I' AND ReceiptAffected = 2; -- Check for specific log entry

    -- You should see one row in ReceiptsLog with OperationType 'I' and ReceiptAffected 2.

    PRINT 'Test Case 1 successful.';

END TRY
BEGIN CATCH
    -- If an error occurs, print error details
    PRINT 'Test Case 1 failed due to an unexpected error:';
    PRINT 'Error Number: ' + CAST(ERROR_NUMBER() AS VARCHAR); -- Display error number [13]
    PRINT 'Error Message: ' + ERROR_MESSAGE(); -- Display error message [13]
    -- THROW; -- Re-throw the error if needed
END CATCH;

-- Rollback the transaction to undo all changes made during the test [9, 12]
-- This removes the inserted rows from Transaction_Table and the log entry from ReceiptsLog.
IF @@TRANCOUNT > 0 -- Check if a transaction is active [14]
    ROLLBACK TRAN; -- Undo changes [9, 12]

PRINT '--- Test Case 1 finished (transaction rolled back) ---';
GO

-- Test Case 2: Operation Blocked by Daily Limit (Attempting to insert 6 rows)
-- Assumes valid Street_Number_PK exists in Shop_Table (e.g., 1)
-- Assumes Receipt_No values 103 through 108 are not already used and fit tinyint range (0-255) [3].
BEGIN TRAN; -- Start a transaction to isolate the test [9, 11, 12]

BEGIN TRY
    PRINT '--- Starting Test Case 2: Operation Blocked (Attempting to insert 6 rows) ---';

    -- Attempt to insert 6 sample rows into Transaction_Table 
    -- This should exceed the daily limit of 5 affected rows and trigger the error from trg8.
    INSERT INTO dbo.Transaction_Table (Receipt_No, Member_Name, Date_of_Receipt, Time_of_Receipt, Total_Payment, Total_Tax, Street_Number_PK)
    VALUES
        (213, 'Customer C', '2023-10-28', '12:00:00', 10.00, 1.00, 1),
        (214, 'Customer D', '2023-10-28', '13:00:00', 15.00, 1.50, 1),
        (215, 'Customer E', '2023-10-28', '14:00:00', 20.00, 2.00, 1),
        (216, 'Customer F', '2023-10-28', '15:00:00', 25.00, 2.50, 1),
        (217, 'Customer G', '2023-10-28', '16:00:00', 30.00, 3.00, 1),
        (218, 'Customer H', '2023-10-28', '17:00:00', 35.00, 3.50, 1); -- This 6th row makes @AffectedRows = 6

    -- If execution reaches here, the THROW statement in the trigger did NOT fire, which is unexpected.
    PRINT 'Error: Insert statement did NOT fail as expected!';

END TRY
BEGIN CATCH
    -- Catch block for handling the expected error
    
    IF ERROR_NUMBER() = 50002 AND ERROR_STATE() = 1
    BEGIN
        PRINT 'Successfully caught expected error from trg8:';
        PRINT 'Error Number: ' + CAST(ERROR_NUMBER() AS VARCHAR); 
        PRINT 'Error State: ' + CAST(ERROR_STATE() AS VARCHAR);
        PRINT 'Error Message: ' + ERROR_MESSAGE();
        
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

-- Verify that no rows were inserted into Transaction_Table (due to the expected THROW and implicit rollback)
PRINT 'Checking Transaction_Table for inserted rows...';
-- Use a WHERE clause that matches the inserted test data
SELECT Receipt_No, Member_Name, Date_of_Receipt
FROM dbo.Transaction_Table
WHERE Receipt_No BETWEEN 213 AND 218; -- Should return 0 rows

-- Verify that no log entry was created by trg8 (because the operation was blocked)
PRINT 'Checking ReceiptsLog for today''s entry...';
SELECT LogDate, LoginName, UserName, OperationType, ReceiptAffected
FROM ReceiptsLog
WHERE LogDate = CAST(GETDATE() AS DATE)
  AND OperationType = 'I' -- Check for insert logs
  AND ReceiptAffected = 6; -- Check for the specific attempt

-- Should return 0 rows in ReceiptsLog.

-- Rollback the transaction to undo everything, just in case [9, 12].
-- The error caught by TRY...CATCH usually handles the statement rollback,
-- but an explicit ROLLBACK TRAN ensures the entire test transaction is cleaned up.
IF @@TRANCOUNT > 0 -- Check if a transaction is active [14]
    ROLLBACK TRAN; -- Undo changes [9, 12]

PRINT '--- Test Case 2 finished (transaction rolled back) ---';
GO