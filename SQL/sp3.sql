CREATE OR ALTER PROC sp3
    @inputDate DATE
AS
BEGIN
    BEGIN TRY
        IF @inputDate > GETDATE()
        BEGIN
            THROW 50001, 'Invalid receipt date to compare!', 1;
        END

        SELECT 
            Receipt_No, 
            Date_of_Receipt, 
            Total_Payment
        FROM Transaction_Table
        WHERE Date_of_Receipt = @inputDate;

        SELECT COUNT(*) AS TotalReceipts
        FROM Transaction_Table
        WHERE Date_of_Receipt = @inputDate;
    END TRY
    BEGIN CATCH
        SELECT ERROR_NUMBER() AS ErrorNumber, ERROR_MESSAGE() AS ErrorMessage;
    END CATCH
END;
GO

-- TESTING SCRIPT
EXEC sp3 @inputDate = '2023-10-27';
