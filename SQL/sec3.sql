USE [EERDBS-3];
GO

-- Create login g3 and user u3
CREATE LOGIN g3 WITH PASSWORD = 'g3', DEFAULT_DATABASE = [EERDBS-3], CHECK_EXPIRATION = OFF, CHECK_POLICY = OFF;
GO

CREATE USER u3 FOR LOGIN g3;
GO

-- Grant SELECT on Transaction_Table
GRANT SELECT ON dbo.Transaction_Table TO u3;
GO

-- Grant INSERT permission (restricted by trigger)
GRANT INSERT ON dbo.Transaction_Table TO u3;
GO

-- Grant UPDATE only on Date_of_Receipt
GRANT UPDATE ON dbo.Transaction_Table (Date_of_Receipt) TO u3;
GO

-- DENY DELETE to u3 (even if role is added later)
DENY DELETE ON dbo.Transaction_Table TO u3;
GO

-- Create a trigger to restrict INSERT/UPDATE where Total_Payment > $25
CREATE OR ALTER TRIGGER trg_Restrict_u3_InsertUpdate
ON dbo.Transaction_Table
AFTER INSERT, UPDATE
AS
BEGIN
    IF EXISTS (
        SELECT 1 FROM inserted WHERE Total_Payment > 25
    )
    BEGIN
        RAISERROR ('Total_Payment must not exceed $25.', 16, 1);
        ROLLBACK TRANSACTION;
    END
END;
GO

-- TESTING BLOCK
PRINT 'Testing permissions for u3';
EXECUTE AS USER = 'u3';
GO

-- Test SELECT
PRINT 'Testing SELECT on Transaction_Table:';
BEGIN TRY
    SELECT TOP 1 * FROM dbo.Transaction_Table;
    PRINT 'SELECT succeeded.';
END TRY
BEGIN CATCH
    PRINT 'SELECT failed.';
    PRINT ERROR_MESSAGE();
END CATCH
GO

-- Test INSERT (valid)
PRINT 'Testing INSERT ≤ $25:';
BEGIN TRY
    BEGIN TRAN;
    INSERT INTO dbo.Transaction_Table (Receipt_No, Total_Payment, Date_of_Receipt)
    VALUES (9991, 24.99, GETDATE());
    PRINT 'INSERT succeeded (under $25)';
    ROLLBACK;
END TRY
BEGIN CATCH
    PRINT 'INSERT failed';
    PRINT ERROR_MESSAGE();
    IF @@TRANCOUNT > 0 ROLLBACK;
END CATCH
GO

-- Test INSERT (invalid)
PRINT 'Testing INSERT > $25:';
BEGIN TRY
    BEGIN TRAN;
    INSERT INTO dbo.Transaction_Table (Receipt_No, Total_Payment, Date_of_Receipt)
    VALUES (9992, 30.00, GETDATE());
    PRINT 'INSERT succeeded (should NOT happen)';
    ROLLBACK;
END TRY
BEGIN CATCH
    PRINT 'INSERT failed as expected';
    PRINT ERROR_MESSAGE();
    IF @@TRANCOUNT > 0 ROLLBACK;
END CATCH
GO

-- Test UPDATE Date_of_Receipt (valid)
PRINT 'Testing UPDATE ≤ $25:';
BEGIN TRY
    BEGIN TRAN;
    UPDATE dbo.Transaction_Table 
    SET Date_of_Receipt = GETDATE()
    WHERE Total_Payment <= 25;
    PRINT 'UPDATE Date_of_Receipt succeeded (under $25)';
    ROLLBACK;
END TRY
BEGIN CATCH
    PRINT 'UPDATE failed';
    PRINT ERROR_MESSAGE();
    IF @@TRANCOUNT > 0 ROLLBACK;
END CATCH
GO

-- Test DELETE (should fail)
PRINT 'Testing DELETE:';
BEGIN TRY
    BEGIN TRAN;
    DELETE FROM dbo.Transaction_Table WHERE Receipt_No = 9991;
    PRINT 'DELETE succeeded (should NOT happen)';
    ROLLBACK;
END TRY
BEGIN CATCH
    PRINT 'DELETE failed as expected';
    PRINT ERROR_MESSAGE();
    IF @@TRANCOUNT > 0 ROLLBACK;
END CATCH
GO

REVERT;
GO
