USE [EERDBS-3];
GO


CREATE OR ALTER TRIGGER trg2
ON dbo.Transaction_Table
AFTER INSERT, UPDATE
AS
BEGIN
  
    IF EXISTS (
        SELECT T.Date_of_Receipt
        FROM dbo.Transaction_Table AS T -- Query the main table to get the current state counts
        WHERE T.Date_of_Receipt IN (SELECT Date_of_Receipt FROM Inserted) -- Limit the check to dates affected by the current operation
        GROUP BY T.Date_of_Receipt -- Group by date to count per day
        HAVING COUNT(*) > 3 -- Identify dates with more than 3 receipts
    )
    BEGIN
        
        DECLARE @OperationType VARCHAR(10);
        -- Check if the Deleted table has any rows
  
        IF EXISTS (SELECT 1 FROM Deleted)
            SET @OperationType = 'UPDATE';
        ELSE -- If Deleted is empty, it was an INSERT
            SET @OperationType = 'INSERT';

        -- Build the custom error message string.
       
        DECLARE @ErrorMessage NVARCHAR(200);
        SET @ErrorMessage = 'Sorry, no more than three receipts with the same receipt date is allowed. Your operation of ' + @OperationType + ' has been reset.';
       
        THROW 50002, @ErrorMessage, 1; -- The third parameter (state) is optional but often set to 1

        ROLLBACK TRAN;
    END;
    -- If the IF EXISTS condition is false, the rule is not violated, and the trigger simply finishes.
END;
GO

BEGIN TRAN;
-- Insert 3 receipts for a specific date
INSERT INTO Transaction_Table (Receipt_No, Member_Name, Date_of_Receipt, Time_of_Receipt, Total_Payment, Total_Tax) VALUES (221,'member 1', '2023-11-15', '08:00:00', 50.00, 3.00);
INSERT INTO Transaction_Table (Receipt_No, Member_Name, Date_of_Receipt, Time_of_Receipt, Total_Payment, Total_Tax) VALUES (222,'memebr 2', '2023-11-15', '09:00:00', 150.00, 9.00);
INSERT INTO Transaction_Table (Receipt_No, Member_Name, Date_of_Receipt, Time_of_Receipt, Total_Payment, Total_Tax) VALUES (223,'member 3', '2023-11-15', '10:00:00', 250.00, 15.00);

    
-- SELECT * FROM Transaction_Table WHERE Date_of_Receipt = '2023-11-15'; -- Optional: Verify setup
COMMIT TRAN; -- Make the setup permanent

-- Test Case 1: Attempt to insert a 4th receipt for the same date (should fail)
BEGIN TRAN;
PRINT 'Attempting to insert a 4th receipt for 2023-11-15...';
INSERT INTO Transaction_Table (Receipt_No, Member_Name, Date_of_Receipt, Time_of_Receipt, Total_Payment, Total_Tax) VALUES ('224','member 4', '2023-11-15','08:00:00', 90.00, 4.00); -- This should trigger the error
-- SELECT * FROM Transaction_Table WHERE Date_of_Receipt = '2023-11-15'; -- Run this after the INSERT to see if it was rolled back
ROLLBACK TRAN; -- Undo the failed INSERT attempt and the setup if it wasn't committed
