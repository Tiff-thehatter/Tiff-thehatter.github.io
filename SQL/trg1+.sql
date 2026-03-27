
USE [EERDBS-3];
GO

-- Create the trigger trg1
-- This trigger prevents updates or deletions on receipts with Total_Payment >= $100


CREATE OR ALTER TRIGGER trg1
ON dbo.Transaction_Table 
AFTER UPDATE, DELETE -- The trigger fires after UPDATE or DELETE operations 
AS
BEGIN
    SET NOCOUNT ON; -- Prevents extra result sets from interfering with application logic 

    -- Check if any affected row has Total_Payment >= $100

    IF EXISTS (SELECT 1 FROM Inserted WHERE Total_Payment >= 100)
       OR EXISTS (SELECT 1 FROM Deleted WHERE Total_Payment >= 100)
    BEGIN
        DECLARE @OperationType VARCHAR(10);

    
        IF EXISTS (SELECT 1 FROM Inserted) AND EXISTS (SELECT 1 FROM Deleted)
            SET @OperationType = 'UPDATE';
        -- DELETE operations result in rows only in Deleted
        ELSE IF NOT EXISTS (SELECT 1 FROM Inserted) AND EXISTS (SELECT 1 FROM Deleted)
            SET @OperationType = 'DELETE';
        -- INSERT operations would result in rows only in Inserted, but this trigger is only on UPDATE, DELETE.

        -- Construct the custom error message as requested
        DECLARE @ErrorMessage NVARCHAR(200);
        SET @ErrorMessage = 'Sorry, no receipts of $100 or higher can be updated or deleted. Your operation of ' + @OperationType + ' has been reset.';

        -- Roll back the transaction to prevent the operation from completing 
        ROLLBACK TRAN;

        -- Throw a custom error message to the user 
        -- Using a custom error number >= 50000
        THROW 50001, @ErrorMessage, 1;

    END;
END;
GO



PRINT '--- Starting Trigger Testing with single BEGIN TRAN/ROLLBACK TRAN ---';
BEGIN TRAN; -- Start a transaction for testing 

-- Insert test data into Transaction_Table
-- Need values for: Receipt_No (PK), Date_of_Receipt, Time_of_Receipt, Total_Payment, Total_Tax
-- Assuming Receipt_No is not an IDENTITY column and we need to provide values.
-- Use values outside the typical range to avoid conflicts with existing data.
INSERT INTO dbo.Transaction_Table (Receipt_No, Member_Name, Date_of_Receipt, Time_of_Receipt, Total_Payment, Total_Tax)
VALUES
    (221,'member 1', '2023-10-27', '08:00:00', 50.00, 3.00), -- Receipt < $100
    (222,'memebr 2', '2023-10-27', '09:00:00', 150.00, 9.00), -- Receipt >= $100
    (223,'member 3', '2023-10-27', '10:00:00', 250.00, 15.00); -- Another Receipt >= $100
PRINT 'Inserted test data (Receipts 221, 222, 223) within transaction.';

-- Test Case 1: Attempt to UPDATE a receipt >= $100 (Receipt_No 222)
PRINT 'Attempting to UPDATE Receipt 222 (150.00) to 160.00 (Expected: Blocked by trigger)';
UPDATE dbo.Transaction_Table
SET Total_Payment = 160.00
WHERE Receipt_No = 222;
-- This operation is expected to fire the trigger, which will ROLLBACK the transaction and THROW an error.
-- Statements after this line might not execute depending on how the THROW error is handled.

-- Test Case 2: Attempt to DELETE a receipt >= $100 (Receipt_No 223)
-- This will only be reached if Test Case 1 did not fire the trigger and roll back.
PRINT 'Attempting to DELETE Receipt 223 (250.00) (Expected: Blocked by trigger, assuming previous statements did not roll back)';
DELETE FROM dbo.Transaction_Table WHERE Receipt_No = 223;
