-- sp8.sql

-- Create or alter the stored procedure sp8
CREATE OR ALTER PROC dbo.sp8 (
    @ReceiptNo INT,
    @ItemID INT, 
    @NewQuantity INT,
    @UpdatedTotalAmount MONEY OUTPUT
)
AS
BEGIN
    -- Declare a variable for the initial quantity for validation
    DECLARE @InitialQuantity INT;
    DECLARE @RowCount INT;

    -- Begin a transaction to ensure atomicity
    BEGIN TRANSACTION;

    BEGIN TRY
        -- Check if the specified item exists for the given receipt and store the initial quantity
        SELECT @InitialQuantity = Quantity
        FROM dbo.Item_Table
        WHERE Receipt_No = @ReceiptNo AND ItemID = @ItemID;

        -- Check if the item was found
        IF @InitialQuantity IS NULL
        BEGIN
            -- Item not found, rollback transaction and return error code 0
            IF @@TRANCOUNT > 0
                ROLLBACK TRANSACTION;
            RETURN 0;
        END;

        -- Validate the new quantity (must be a positive integer)
        IF @NewQuantity <= 0
        BEGIN
            -- Invalid quantity, rollback transaction and return error code 0
            IF @@TRANCOUNT > 0
                ROLLBACK TRANSACTION;
            RETURN 0;
        END;

        -- Update the quantity of the specific item in the receipt
        UPDATE dbo.Item_Table
        SET Quantity = @NewQuantity
        WHERE Receipt_No = @ReceiptNo AND ItemID = @ItemID;

        -- Check if the update was successful
        SET @RowCount = @@ROWCOUNT;
        IF @RowCount = 0
        BEGIN
            -- Update failed (though the item existed), rollback and return error code 0
            IF @@TRANCOUNT > 0
                ROLLBACK TRANSACTION;
            RETURN 0;
        END;

        -- Calculate the updated total amount for the receipt
        SELECT @UpdatedTotalAmount = SUM(Quantity * Price) 
        FROM dbo.Item_Table
        WHERE Receipt_No = @ReceiptNo;

        -- Commit the transaction as the update was successful
        COMMIT TRANSACTION;

        -- Return success code 1
        RETURN 1;
    END TRY
    BEGIN CATCH
        -- If any error occurred, rollback the transaction
        IF @@TRANCOUNT > 0
            ROLLBACK TRANSACTION;

        -- a custom error or just return failure code 0
        -- THROW 50003, 'Failed to update line item quantity!', 1;
        RETURN 0;
    END CATCH;
END;
GO

-- Testing script for sp8
BEGIN TRAN; -- Begin a transaction to allow rollback of test data changes

DECLARE @TestReceiptNo INT = 123; -- Replace with an existing Receipt_No in your Item_Table
DECLARE @TestItemID INT = 1;    -- Replace with an existing ItemID for the above Receipt_No
DECLARE @NewQuantityToSet INT = 2;
DECLARE @CurrentUpdatedTotal MONEY;
DECLARE @ReturnCode INT;

-- Test case 1: Successful update
PRINT '--- Test Case 1: Successful Update ---';
EXEC @ReturnCode = dbo.sp8
    @ReceiptNo = @TestReceiptNo,
    @ItemID = @TestItemID,
    @NewQuantity = @NewQuantityToSet,
    @UpdatedTotalAmount = @CurrentUpdatedTotal OUTPUT;

IF @ReturnCode = 1
    PRINT 'Update successful. Updated Total Amount: ' + CAST(@CurrentUpdatedTotal AS VARCHAR);
ELSE
    PRINT 'Update failed.';
GO