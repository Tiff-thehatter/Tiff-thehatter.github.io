-- sec2.sql

USE [EERDBS-3];
GO

-- Create a new SQL Server authenticated login with passwd g2
CREATE LOGIN g2 WITH PASSWORD = 'g2', DEFAULT_DATABASE = [EERDBS-3], CHECK_EXPIRATION = OFF, CHECK_POLICY = OFF;
GO

-- Create a new user u2 in DB for g2
CREATE USER u2 FOR LOGIN g2;
GO

-- Grant select permission to u2 on Shop_Table
GRANT SELECT ON dbo.Shop_Table TO u2;
GO

-- Grant select permission to u2 on Transaction_Table
GRANT SELECT ON dbo.Transaction_Table TO u2;
GO

-- Deny select permission to u2 on Item_Table
DENY SELECT ON dbo.Item_Table TO u2;
GO

-- Grant execute permission to u2 on the udf7 function
GRANT EXECUTE ON dbo.udf7 TO u2;
GO


-- Grant execute permision on udf1&2
GRANT EXECUTE ON dbo.udf1 TO u2;
GRANT EXECUTE ON dbo.udf2 TO u2;
GO

-- Testing script 
USE [EERDBS-3];
GO

PRINT 'Testing user u2:';
GO

-- Test user u2
EXECUTE AS USER = 'u2';
GO

-- Test SELECT on allowed tables
PRINT 'Testing SELECT on Shop_Table:';
BEGIN TRY
    SELECT * FROM dbo.Shop_Table;
    PRINT 'SELECT on Shop_Table succeeded.';
END TRY
BEGIN CATCH
    PRINT 'SELECT on Shop_Table failed.';
END CATCH
GO

PRINT 'Testing SELECT on Transaction_Table:';
BEGIN TRY
    SELECT * FROM dbo.Transaction_Table;
    PRINT 'SELECT on Transaction_Table succeeded.';
END TRY
BEGIN CATCH
    PRINT 'SELECT on Transaction_Table failed.';
END CATCH
GO

-- Test SELECT on the denied table (Item_Table)
PRINT 'SELECT on Item_Table failed.';
BEGIN TRY
    SELECT * FROM dbo.Item_Table;
    PRINT 'SELECT on Item_Table succeeded.';
END TRY
BEGIN CATCH
    PRINT 'SELECT on Item_Table failed as expected.';
    PRINT ERROR_MESSAGE();
END CATCH
GO

-- Test EXECUTE on allowed functions
PRINT 'Testing EXECUTE on dbo.udf7:';
BEGIN TRY
    SELECT * FROM dbo.udf7 (10, 100); -- Min and Max
    PRINT 'EXECUTE on dbo.udf7 succeeded.';
END TRY
BEGIN CATCH
    PRINT 'EXECUTE on dbo.udf7 failed.';
    PRINT ERROR_MESSAGE();
END CATCH
GO

-- Test EXECUTE on functions
PRINT 'Testing EXECUTE on dbo.udf1:';
BEGIN TRY
    -- udf1-scalar function
    SELECT dbo.udf1() AS TotalExpense;
    PRINT 'EXECUTE on dbo.udf1 succeeded.';
END TRY
BEGIN CATCH
    PRINT 'EXECUTE on dbo.udf1 failed.';
    PRINT ERROR_MESSAGE();
END CATCH
GO

PRINT 'Testing EXECUTE on dbo.udf2:';
BEGIN TRY
    -- udf2-table function
    SELECT * FROM dbo.udf2();
    PRINT 'EXECUTE on dbo.udf2 succeeded.';
END TRY
BEGIN CATCH
    PRINT 'EXECUTE on dbo.udf2 failed.';
    PRINT ERROR_MESSAGE();
END CATCH
GO

-- Test INSERT (should fail)
PRINT 'INSERT into Shop_Table failed:';
BEGIN TRY
    BEGIN TRAN;
    INSERT INTO dbo.Shop_Table (Shop_Name) VALUES ('Test Shop');
    PRINT 'INSERT on Shop_Table succeeded.';
    ROLLBACK TRAN;
END TRY
BEGIN CATCH
    PRINT 'INSERT on Shop_Table failed as expected.';
    PRINT ERROR_MESSAGE();
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
END CATCH
GO

-- Test UPDATE (should fail)
PRINT 'Testing UPDATE on Shop_Table failed';
BEGIN TRY
    BEGIN TRAN;
    UPDATE dbo.Shop_Table SET Shop_Name = 'Updated Shop' WHERE Shop_Name = 'Test Shop';
    PRINT 'UPDATE on Shop_Table succeeded.';
    ROLLBACK TRAN;
END TRY
BEGIN CATCH
    PRINT 'UPDATE on Shop_Table failed as expected.';
    PRINT ERROR_MESSAGE();
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
END CATCH
GO

-- Test DELETE (should fail)
PRINT 'Testing DELETE from Shop_Table failed.:';
BEGIN TRY
    BEGIN TRAN;
    DELETE FROM dbo.Shop_Table WHERE Shop_Name = 'Test Shop';
    PRINT 'DELETE from Shop_Table succeeded.';
    ROLLBACK TRAN;
END TRY
BEGIN CATCH
    PRINT 'DELETE from Shop_Table failed as expected.';
    PRINT ERROR_MESSAGE();
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
END CATCH
GO

-- Test CREATE TABLE (should fail)
PRINT 'CREATE TABLE failed.';
BEGIN TRY
    BEGIN TRAN;
    CREATE TABLE TestTable_u2 (ID INT);
    PRINT 'CREATE TABLE succeeded';
    ROLLBACK TRAN;
END TRY
BEGIN CATCH
    PRINT 'CREATE TABLE failed.';
    PRINT ERROR_MESSAGE();
    IF @@TRANCOUNT > 0 ROLLBACK TRAN;
END CATCH
GO

-- Revert impersonation
REVERT;
GO