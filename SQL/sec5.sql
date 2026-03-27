-- sec5.sql

USE [EERDBS-3];
GO

-- Create login g5 with password 'g5'
-- Disable password expiration and policy checks for simplicity
CREATE LOGIN g5 WITH PASSWORD = 'g5',
DEFAULT_DATABASE = [EERDBS-3],
CHECK_EXPIRATION = OFF,
CHECK_POLICY = OFF;
GO

-- Step 2: Switch to the target database
-- Database-specific operations require changing the database context
USE [EERDBS-3];
GO

-- Step 3: Create user u5 for login g5 in the current database
-- Use the FOR LOGIN clause to explicitly link the user to the login
CREATE USER u5 FOR LOGIN g5;
GO

-- Step 4: Grant u5 the privilege to create users in the database,
-- and allow u5 to grant this privilege to others.
-- The CREATE USER permission allows creating new users
-- The WITH GRANT OPTION allows the recipient to grant the permission to others
-- Granting database permissions uses the GRANT statement without an ON clause

GRANT CREATE USER TO u5 WITH GRANT OPTION;
GO

-- Manual Testing Block 
-- To test step 4:
-- 1. Impersonate user u5: EXECUTE AS USER = 'u5'; [25, 26]
-- 2. Try creating a new login/user, e.g., CREATE LOGIN test_g WITH PASSWORD='pwd'; CREATE USER test_u FOR LOGIN test_g; -- This should succeed if u5 has necessary permissions
-- 3. Try granting CREATE USER permission from u5 to the new user: GRANT CREATE USER TO test_u; -- This should succeed because u5 was granted WITH GRANT OPTION.
-- 4. Revert impersonation: REVERT; [26]
-- 5. You might need server-level permissions (like ALTER ANY LOGIN) to create the login itself.
-- The prompt specifically asks about creating *users* in the database, which CREATE USER permits.

-- Step 5: Withdraw the ability for u5 to grant CREATE USER permission to others.
-- u5 should still be able to create users, but users created by u5 cannot grant this permission further.
-- Use REVOKE GRANT OPTION FOR to remove the ability to grant a permission
REVOKE GRANT OPTION FOR CREATE USER FROM u5 CASCADE;
GO

-- Manual Testing Block
-- To test step 5:
-- 1. Impersonate user u5: EXECUTE AS USER = 'u5';
-- 2. Try creating another new login/user (e.g., test_g2, test_u2): CREATE LOGIN test_g2 WITH PASSWORD='pwd2'; CREATE USER test_u2 FOR LOGIN test_g2; -- This should still succeed as u5 retains CREATE USER permission.
-- 3. Try granting CREATE USER permission *from u5* to the new user: GRANT CREATE USER TO test_u2; -- This should now fail because the GRANT OPTION has been revoked from u5.
-- 4. Revert impersonation: REVERT;

-- Step 6: Revoke u5's privilege to create new users.
-- Use the REVOKE statement to remove the CREATE USER permission from u5 [18, 24].
REVOKE CREATE USER FROM u5 CASCADE;
GO

-- Manual Testing Block 
-- To test step 6:
-- 1. Impersonate user u5: EXECUTE AS USER = 'u5';
-- 2. Try creating another new login/user (e.g., test_g3, test_u3): CREATE LOGIN test_g3 WITH PASSWORD='pwd3'; CREATE USER test_u3 FOR LOGIN test_g3; -- This should now fail because u5's CREATE USER permission has been revoked.
-- 3. Revert impersonation: REVERT;

