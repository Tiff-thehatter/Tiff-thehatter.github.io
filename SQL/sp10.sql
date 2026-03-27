CREATE OR ALTER PROCEDURE sp10
AS
BEGIN
    SET NOCOUNT ON;

    -- Declare variables to store the results
    DECLARE @LargestReceiptNo INT, @LargestReceiptDate DATE, @LargestTotalPayment DECIMAL(10, 2);
    DECLARE @MostLineItemsReceiptNo INT, @MostLineItems INT, @MemberName VARCHAR(50);
    DECLARE @MostFrequentItemID INT, @MostFrequentItemDescription VARCHAR(50), @MostFrequentItemUnitPrice DECIMAL(10, 2), @MostFrequentItemCount INT;

    -- Receipt with the largest total amount
    SELECT TOP 1 @LargestReceiptNo = Receipt_No,
                   @LargestReceiptDate = Date_of_Receipt,
                   @LargestTotalPayment = Total_Payment
    FROM Receipts
    ORDER BY Total_Payment DESC;

    IF @@ROWCOUNT > 0
        BEGIN
            SELECT @LargestReceiptDate = FORMAT(@LargestReceiptDate, 'MM/dd/yyyy');
            PRINT 'The receipt with the largest total amount is receipt ' + CAST(@LargestReceiptNo AS VARCHAR(10)) +
                  ' on ' + @LargestReceiptDate + ' with a total amount of $' + CAST(ROUND(@LargestTotalPayment, 2) AS VARCHAR(20));
        END
    ELSE
        BEGIN
            PRINT 'No receipts found.';
        END

    -- Receipt with the most line items
    SELECT TOP 1 @MostLineItemsReceiptNo = r.Receipt_No,
                   @MostLineItems = COUNT(*)
    FROM Receipts r
    JOIN Items i ON r.Receipt_No = i.Receipt_No
    GROUP BY r.Receipt_No
    ORDER BY COUNT(*) DESC;

    IF @@ROWCOUNT > 0
        BEGIN
            SELECT @MemberName = Member_Name
            FROM Receipts
            WHERE Receipt_No = @MostLineItemsReceiptNo;

            PRINT 'The receipt with the most line items is receipt ' + CAST(@MostLineItemsReceiptNo AS VARCHAR(10)) +
                  ' with ' + CAST(@MostLineItems AS VARCHAR(10)) + ' line items, whose owner is ' + @MemberName + '.';
        END
    ELSE
        BEGIN
            PRINT 'No receipts found.';
        END

    -- Item that is most frequently purchased
    SELECT TOP 1 @MostFrequentItemID = Item_ID,
                   @MostFrequentItemCount = COUNT(*)
    FROM Items
    GROUP BY Item_ID
    ORDER BY COUNT(*) DESC;

    IF @@ROWCOUNT > 0
        BEGIN
            SELECT @MostFrequentItemDescription = Item_Description,
                   @MostFrequentItemUnitPrice = Unit_Price
            FROM Items
            WHERE Item_ID = @MostFrequentItemID;

            PRINT 'The item that is most frequently purchased is item ' + CAST(@MostFrequentItemID AS VARCHAR(10)) +
                  ', ' + @MostFrequentItemDescription + ', $' + CAST(ROUND(@MostFrequentItemUnitPrice, 2) AS VARCHAR(20)) +
                  ', in ' + CAST(@MostFrequentItemCount AS VARCHAR(10)) + ' receipts';
        END
    ELSE
        BEGIN
            PRINT 'No items found.';
        END
END;
GO

-- Testing script

CREATE TABLE IF NOT EXISTS Receipts (
    Receipt_No INT PRIMARY KEY,
    Member_Name VARCHAR(50),
    Date_of_Receipt DATE,
    Total_Payment DECIMAL(10, 2)
);

CREATE TABLE IF NOT EXISTS Items (
    Item_ID INT,
    Receipt_No INT,
    Item_Description VARCHAR(50),
    Unit_Price DECIMAL(10, 2),
    FOREIGN KEY (Receipt_No) REFERENCES Receipts(Receipt_No)
);


-- Clear existing data
DELETE FROM Items;
DELETE FROM Receipts;

-- Insert sample data
INSERT INTO Receipts (Receipt_No, Member_Name, Date_of_Receipt, Total_Payment) VALUES
(1, 'John Doe', '2024-01-10', 100.00),
(2, 'Jane Smith', '2024-01-15', 250.50),
(3, 'Peter Jones', '2024-02-01', 150.75),
(4, 'John Doe', '2024-02-10', 300.00);

INSERT INTO Items (Item_ID, Receipt_No, Item_Description, Unit_Price) VALUES
(101, 1, 'Laptop', 800.00),
(102, 1, 'Mouse', 25.00),
(201, 2, 'Keyboard', 75.00),
(202, 2, 'Monitor', 200.00),
(203, 2, 'Webcam', 50.50),
(301, 3, 'Printer', 120.00),
(302, 3, 'Cable', 30.75),
(401, 4, 'Software', 100.00),
(402, 4, 'License', 200.00),
(101, 2, 'Laptop', 800.00);  -- Item 101 appears in two receipts


-- Execute the stored procedure
EXEC sp10;
