-- sp6.sql

-- Create a user-defined table type for the items in the receipt
IF TYPE_ID(N'dbo.ItemTableType') IS NULL
CREATE TYPE dbo.ItemTableType AS TABLE
(
    Quantity INT,
    Item_Description VARCHAR(255) NOT NULL,
    Unit_Price DECIMAL(10, 2) NOT NULL
);
GO

-- Create or alter the stored procedure sp6
CREATE OR ALTER PROC dbo.sp6 (
    @Receipt_No TINYINT,
    @Member_Name NVARCHAR(255),
    @sales_associate NVARCHAR(255),
    @Date_of_Receipt DATE,
    @Time_of_Receipt TIME,
    @Total_Payment MONEY,
    @Payment_Method NVARCHAR(50),
    @Total_Tax MONEY,
    @Street_Number_PK SMALLINT,
    @Items dbo.ItemTableType READONLY -- Table-valued parameter for items
)
AS
BEGIN
    -- Begin a transaction to ensure atomicity
    BEGIN TRANSACTION;

    BEGIN TRY
        -- Insert into Transaction_Table
        INSERT INTO Transaction_Table (
            Receipt_No,
            Member_Name,
            sales_associate,
            Date_of_Receipt,
            Time_of_Receipt,
            Total_Payment,
            Payment_Method,
            Total_Tax,
            Street_Number_PK
        )
        VALUES (
            @Receipt_No,
            @Member_Name,
            @sales_associate,
            @Date_of_Receipt,
            @Time_of_Receipt,
            @Total_Payment,
            @Payment_Method,
            @Total_Tax,
            @Street_Number_PK
        );

        -- Insert items into Item_Table using the table-valued parameter
        INSERT INTO Item_Table (
            Quantity,
            Item_Description,
            Unit_Price,
            Receipt_No
        )
        SELECT
            Quantity,
            Item_Description,
            Unit_Price,
            @Receipt_No
        FROM @Items;

        -- Commit the transaction if all inserts are successful
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        -- If any error occurred, rollback the transaction
        IF @@TRANCOUNT > 0
            ROLLBACK TRANSACTION;

        -- Throw a custom error to the caller
        THROW 50002, 'Insert receipt failed!', 1;
        RETURN;
    END CATCH;
END;
GO

-- Test Case for sp6
BEGIN TRY
    -- Declare a table variable of the ItemTableType
    DECLARE @ReceiptItems dbo.ItemTableType;

    -- Insert sample items into the table variable
    INSERT INTO @ReceiptItems (Quantity, Item_Description, Unit_Price)
    VALUES
        (2, 'Laptop', 1200.00),
        (1, 'Mouse', 25.00),
        (3, 'USB Drive', 15.00);

    -- Execute the sp6 stored procedure with sample data
    EXEC dbo.sp6
        @Receipt_No = 12345,
        @Member_Name = 'John Doe',
        @sales_associate = 'Jane Smith',
        @Date_of_Receipt = '2023-10-27',
        @Time_of_Receipt = '10:30:00',
        @Total_Payment = 2480.00,
        @Payment_Method = 'Credit Card',
        @Total_Tax = 248.00,
        @Street_Number_PK = 101,
        @Items = @ReceiptItems;

    -- Print a success message if the procedure completes without error
    PRINT 'Receipt inserted successfully!';
END TRY
BEGIN CATCH
    -- Catch any error thrown by the stored procedure
    PRINT 'Error inserting receipt!';
    PRINT 'Error Number: ' + CONVERT(VARCHAR, ERROR_NUMBER());
    PRINT 'Error Message: ' + ERROR_MESSAGE();
END CATCH;
GO