USE [EERDBS-3];
GO

CREATE OR ALTER TRIGGER trg3
ON dbo.Item_Table
AFTER INSERT, UPDATE
AS
BEGIN
    IF EXISTS (
        SELECT 1
        FROM Inserted i
        JOIN dbo.Transaction_Table t ON i.Receipt_No = t.Receipt_No
        WHERE 
            i.Unit_Price >= 500
            AND t.Date_of_Receipt >= '2025-03-01'
            AND t.Date_of_Receipt < '2025-04-01'
    )
    BEGIN
        ROLLBACK TRAN;
        THROW 50001, 'Blocked: Item >= $500 in receipt dated March 2025.', 1;
        RETURN;
    END
END
GO

BEGIN TRAN;


-- Insert two test receipts into Transaction_Table: one dated before April 2025, one dated in April 2025.
IF NOT EXISTS (SELECT 1 FROM dbo.Transaction_Table WHERE Receipt_No = 1)
INSERT INTO dbo.Transaction_Table (Receipt_No, Member_Name, sales_associate, Date_of_Receipt, Time_of_Receipt, Total_Payment, Payment_Method, Total_Tax, Street_Number_PK)
VALUES (1, 'OldReceiptMember', 'SA_OLD', '2025-03-15', '10:00:00', 100.00, 'Credit', 8.00, 1); -- Receipt dated before April 2025 [20]
IF NOT EXISTS (SELECT 1 FROM dbo.Transaction_Table WHERE Receipt_No = 2)
INSERT INTO dbo.Transaction_Table (Receipt_No, Member_Name, sales_associate, Date_of_Receipt, Time_of_Receipt, Total_Payment, Payment_Method, Total_Tax, Street_Number_PK)
VALUES (2, 'NewReceiptMember', 'SA_NEW', '2025-04-10', '11:00:00', 150.00, 'Cash', 12.00, 1); -- Receipt dated in April 2025 [20]
-- Setup initial items (using unique I_D values) [1]
IF NOT EXISTS (SELECT 1 FROM dbo.Item_Table WHERE I_D = 980)
INSERT INTO dbo.Item_Table (I_D, Item_Description, Unit_Price, Receipt_No) VALUES (980, 'Item for Old Receipt', 10.00, 1); -- Item for Receipt 1 [1]
IF NOT EXISTS (SELECT 1 FROM dbo.Item_Table WHERE I_D = 981)
INSERT INTO dbo.Item_Table (I_D, Item_Description, Unit_Price, Receipt_No) VALUES (981, 'Item for New Receipt', 10.00, 2); -- Item for Receipt 2 [1]

PRINT 'Test Case 1: Attempting INSERT of item (>= $500) into Receipt 2 (dated Apr 2025). Expected: SUCCEED.';
BEGIN TRY
    -- Use a NEW, UNIQUE I_D (e.g., 982)
    IF NOT EXISTS (SELECT 1 FROM dbo.Item_Table WHERE I_D = 982)
INSERT INTO dbo.Item_Table (I_D, Item_Description, Unit_Price, Receipt_No) VALUES (982, 'Violating New Item 1', 550.00, 2); -- Using a unique I_D [1]
    -- If the previous line didn't raise an error, this means trg3 didn't block it (as expected for a receipt in April 2025)
    PRINT 'Result: SUCCESS - Operation was correctly allowed (not blocked by trg3).'; -- Changed success message
END TRY
BEGIN CATCH
    -- If an error occurred here, it means trg3 *did* block it, which is unexpected for this test case
    IF ERROR_NUMBER() = 50001
        PRINT 'Result: FAILURE - Unexpectedly blocked by trg3: ' + ERROR_MESSAGE(); -- Changed failure message
    ELSE
        PRINT 'Result: FAILURE - An unexpected error occurred: ' + ERROR_MESSAGE();
END CATCH;


PRINT 'Test Case 2: Attempting INSERT of item (>= $500) into Receipt 1 (dated Mar 2025). Expected: BLOCKED by trg3.'; -- Changed expected outcome
BEGIN TRY
    -- Use a NEW, UNIQUE I_D (e.g., 983)
    IF NOT EXISTS (SELECT 1 FROM dbo.Item_Table WHERE I_D = 983)
INSERT INTO dbo.Item_Table (I_D, Item_Description, Unit_Price, Receipt_No) VALUES (983, 'Allowed New Item', 550.00, 1); -- Using a unique I_D [1]
    -- If the previous line didn't raise an error, it means trg3 *didn't* block it, which is unexpected for this test case
    PRINT 'Result: FAILURE - Operation was unexpectedly allowed (not blocked by trg3).'; -- Changed failure message
END TRY
BEGIN CATCH
    -- If an error occurred, it was expected if trg3 blocked it
    IF ERROR_NUMBER() = 50001
        PRINT 'Result: SUCCESS - Operation was correctly blocked by trg3.'; -- Changed success message
    ELSE
        PRINT 'Result: FAILURE - An unexpected error occurred: ' + ERROR_MESSAGE();
END CATCH;
IF NOT EXISTS (SELECT 1 FROM dbo.Transaction_Table WHERE Receipt_No = 1)
INSERT INTO dbo.Transaction_Table (Receipt_No, Member_Name, sales_associate, Date_of_Receipt, Time_of_Receipt, Total_Payment, Payment_Method, Total_Tax, Street_Number_PK)

VALUES (1, 'OldReceiptMember', 'SA_OLD', '2025-03-15', '10:00:00', 100.00, 'Credit', 8.00, 1); -- Receipt dated before April 2025

IF NOT EXISTS (SELECT 1 FROM dbo.Transaction_Table WHERE Receipt_No = 2)
INSERT INTO dbo.Transaction_Table (Receipt_No, Member_Name, sales_associate, Date_of_Receipt, Time_of_Receipt, Total_Payment, Payment_Method, Total_Tax, Street_Number_PK)

VALUES (2, 'NewReceiptMember', 'SA_NEW', '2025-04-10', '11:00:00', 150.00, 'Cash', 12.00, 1); -- Receipt dated in April 2025

-- Insert two initial test items into Item_Table. These will be used for UPDATE test cases.

IF NOT EXISTS (SELECT 1 FROM dbo.Item_Table WHERE I_D = 980)
INSERT INTO dbo.Item_Table (I_D, Item_Description, Unit_Price, Receipt_No) VALUES (980, 'Item for Old Receipt', 10.00, 1); -- Item for Receipt 1
IF NOT EXISTS (SELECT 1 FROM dbo.Item_Table WHERE I_D = 981)
INSERT INTO dbo.Item_Table (I_D, Item_Description, Unit_Price, Receipt_No) VALUES (981, 'Item for New Receipt', 10.00, 2); -- Item for Receipt 2

-- Test Case 1: Insert a NEW item with price >= $500 into Receipt 2 (dated Apr 2025).
BEGIN TRY
    IF NOT EXISTS (SELECT 1 FROM dbo.Item_Table WHERE I_D = 980)
INSERT INTO dbo.Item_Table (I_D, Item_Description, Unit_Price, Receipt_No) VALUES (980, 'Violating New Item 1', 550.00, 2);
END TRY
BEGIN CATCH
    -- Check if the error caught is the custom error thrown by trg3
    IF ERROR_NUMBER() = 50001
        PRINT 'Result: SUCCESS - Operation was correctly blocked by trg3.';
    ELSE
        PRINT 'Result: FAILURE - An unexpected error occurred: ' + ERROR_MESSAGE();
END CATCH;

-- Test Case 2: Insert a NEW item with price >= $500 into Receipt 1 (dated Mar 2025).
PRINT 'Test Case 2: Attempting INSERT of item (>= $500) into Receipt 1 (dated Mar 2025). Expected: SUCCEED.';
BEGIN TRY
    -- Use a unique I_D (104).
    IF NOT EXISTS (SELECT 1 FROM dbo.Item_Table WHERE I_D = 981)
INSERT INTO dbo.Item_Table (I_D, Item_Description, Unit_Price, Receipt_No) VALUES (981, 'Allowed New Item', 550.00, 1);
    PRINT 'Result: SUCCESS - Operation was correctly allowed.';
END TRY
BEGIN CATCH
    -- If an error occurred, it was unexpected, as the operation should succeed.
    PRINT 'Result: FAILURE - An unexpected error occurred: ' + ERROR_MESSAGE();
END CATCH;

ROLLBACK TRAN;

GO