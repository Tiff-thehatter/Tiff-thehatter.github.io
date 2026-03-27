USE [master];
GO

-- Step 1: Create new login g4
CREATE LOGIN g4 
WITH PASSWORD = 'g4', 
     DEFAULT_DATABASE = [EERDBS-3], 
     CHECK_EXPIRATION = OFF, 
     CHECK_POLICY = OFF;
GO

USE [EERDBS-3];
GO

-- Step 2: Drop user u3 (if it exists) to remap it to g4
-- You may want to back up any permissions first
DROP USER u3;
GO

-- Step 3: Create user u3 again but map it to g4
CREATE USER u3 FOR LOGIN g4;
GO

-- Step 4: Rename u3 to u4
ALTER USER u3 WITH NAME = u4;
GO

-- Step 5: Disable login g3 (so they can no longer connect)
ALTER LOGIN g3 DISABLE;
GO

-- Step 6: Change password of g4
ALTER LOGIN g4 WITH PASSWORD = 'g4-for-u3';
GO

-- Step 7: TESTING block
PRINT '--- Testing impersonation of renamed user and disabled login ---';
GO

-- Test impersonation of u4 (formerly u3)
BEGIN TRY
    EXECUTE AS USER = 'u4';
    PRINT 'Successfully impersonated u4';
    REVERT;
END TRY
BEGIN CATCH
    PRINT 'Failed to impersonate u4';
    PRINT ERROR_MESSAGE();
END CATCH;
GO

-- Test impersonation of g3 (should fail because it's disabled)
BEGIN TRY
    EXECUTE AS LOGIN = 'g3';
    PRINT 'Unexpectedly succeeded impersonating g3 (this is a problem)';
    REVERT;
END TRY
BEGIN CATCH
    PRINT 'Correctly failed to impersonate g3';
    PRINT ERROR_MESSAGE();
END CATCH;
GO

-- Test impersonation of g4 (should succeed with new password)
BEGIN TRY
    EXECUTE AS LOGIN = 'g4';
    PRINT 'Successfully impersonated g4';
    REVERT;
END TRY
BEGIN CATCH
    PRINT 'Failed to impersonate g4';
    PRINT ERROR_MESSAGE();
END CATCH;
GO
