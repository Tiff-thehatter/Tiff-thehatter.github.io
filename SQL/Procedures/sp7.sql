-- sp7.sql

-- Create or alter the stored procedure sp7
CREATE OR ALTER PROC dbo.sp7 (
    @ReceiptNo INT
)
AS
BEGIN
    -- Begin a transaction within the stored procedure
    BEGIN TRANSACTION;

    BEGIN TRY
        -- Delete items associated with the receipt
        DELETE FROM dbo.Item_Table
        WHERE Receipt_No = @ReceiptNo;

        -- Delete the receipt from the Transaction_Table
        DELETE FROM dbo.Transaction_Table
        WHERE Receipt_No = @ReceiptNo;

        -- Commit the transaction if both deletes are successful
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        -- If any error occurred, rollback the transaction
        IF @@TRANCOUNT > 0
            ROLLBACK TRANSACTION;

        -- Throw a custom error to the caller
        THROW 50002, 'Delete receipt failed!', 1;
        RETURN;
    END CATCH;
END;
GO

-- Testing script for sp7
BEGIN TRAN; -- Begin a transaction to allow rollback

BEGIN TRY
    -- Declare a sample Receipt_No to delete
    DECLARE @TestReceiptNo INT = 12345;

    -- Execute the sp7 stored procedure
    EXEC dbo.sp7 @ReceiptNo = @TestReceiptNo;

    -- Print a success message if the procedure completes without error
    PRINT 'Receipt with Receipt_No = ' + CAST(@TestReceiptNo AS VARCHAR) + ' was marked for deletion.';

    -- the receipt and its items would be deleted.
    -- The ROLLBACK TRAN at the end of this script will undo the changes.
END TRY
BEGIN CATCH
    -- Catch any error thrown by the stored procedure
    PRINT 'Error deleting receipt!';
    PRINT 'Error Number: ' + CONVERT(VARCHAR, ERROR_NUMBER());
    PRINT 'Error Message: ' + ERROR_MESSAGE();
END CATCH;

ROLLBACK TRAN; -- Rollback the transaction to undo the deletion
PRINT 'Deletion of receipt (if attempted) has been rolled back.';
GO