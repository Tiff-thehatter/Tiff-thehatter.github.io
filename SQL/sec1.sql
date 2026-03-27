-- sec1.sql

USE [EERDBS-3];
GO

-- Create a new login g1 with passwd g1
CREATE LOGIN g1 WITH PASSWORD = 'g1', DEFAULT_DATABASE = [EERDBS-3], CHECK_EXPIRATION = OFF, CHECK_POLICY = OFF;
GO

-- Create a new user u1 in the EERDBS-3 with login g1
CREATE USER u1 FOR LOGIN g1;
GO

-- Assign the user u1 to the db_owner role to make them a co-owner of the database
ALTER ROLE db_owner ADD MEMBER u1;
GO


USE [EERDBS-3];
GO

-- Test login g1:attempt to create a table
PRINT 'Testing login g1:';
BEGIN TRY
    EXECUTE AS LOGIN = 'g1';
    BEGIN TRAN;
    CREATE TABLE TestTable_g1 (ID INT);
    PRINT 'Login g1 successfully created a table.';
    ROLLBACK TRAN;
END TRY
BEGIN CATCH
    PRINT 'Login g1 does not have sufficient permissions to create a table or impersonation failed.';
END CATCH
REVERT;
GO

-- Test user u1: attempt to create a table
PRINT 'Testing user u1:';
BEGIN TRY
    EXECUTE AS USER = 'u1';
    BEGIN TRAN;
    CREATE TABLE TestTable_u1 (ID INT);
    PRINT 'User u1 successfully created a table.';
    ROLLBACK TRAN;
END TRY
BEGIN CATCH
    PRINT 'User u1 does not have sufficient permissions to create a table or impersonation failed.';
END CATCH
REVERT;
GO

-- Test login g1: Impersonate and attempt to create a user
PRINT 'Testing login g1 ability to create a user:';
BEGIN TRY
    EXECUTE AS LOGIN = 'g1';
    BEGIN TRAN;
    CREATE USER test_user_g1 FOR LOGIN g1;
    PRINT 'Login g1 successfully created a user.';
    DROP USER test_user_g1;
    ROLLBACK TRAN;
END TRY
BEGIN CATCH
    PRINT 'Login g1 does not have sufficient permissions to create a user or impersonation failed.';
END CATCH
REVERT;
GO

-- Test user u1: attempt to create a user
PRINT 'Testing user u1 ability to create a user:';
BEGIN TRY
    EXECUTE AS USER = 'u1';
    BEGIN TRAN;
    CREATE USER test_user_u1 FOR LOGIN g1;
    PRINT 'User u1 successfully created a user.';
    DROP USER test_user_u1;
    ROLLBACK TRAN;
END TRY
BEGIN CATCH
    PRINT 'User u1 does not have sufficient permissions to create a user or impersonation failed.';
END CATCH
REVERT;
GO

-- Test login g1: attempt to alter a property
PRINT 'Testing login g1 ability to alter a database property:';
BEGIN TRY
    EXECUTE AS LOGIN = 'g1';
    ALTER DATABASE [EERDBS-3] MODIFY NAME = [EERDBS-3_RenamedForTest];
    PRINT 'Login g1 successfully altered a database property.';
    ALTER DATABASE [EERDBS-3_RenamedForTest] MODIFY NAME = [EERDBS-3]; -- Rename back
END TRY
BEGIN CATCH
    PRINT 'Login g1 does not have sufficient permissions to alter a database property or impersonation failed.';
END CATCH
REVERT;
GO

-- Test user u1: attempt to alter a property
PRINT 'Testing user u1 ability to alter a database property:';
BEGIN TRY
    EXECUTE AS USER = 'u1';
    ALTER DATABASE [EERDBS-3] MODIFY NAME = [EERDBS-3_RenamedForTest];
    PRINT 'User u1 successfully altered a database property.';
    ALTER DATABASE [EERDBS-3_RenamedForTest] MODIFY NAME = [EERDBS-3]; -- Rename back
END TRY
BEGIN CATCH
    PRINT 'User u1 does not have sufficient permissions to alter a database property or impersonation failed.';
END CATCH
REVERT;
GO