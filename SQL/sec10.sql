-- sec10.sql
USE [EERDBS-3];
GO

-- 1. Create server login and database user
IF NOT EXISTS (SELECT * FROM sys.server_principals WHERE name = 'g10')
  CREATE LOGIN g10 WITH PASSWORD = 'g10';
GO

IF NOT EXISTS (SELECT * FROM sys.database_principals WHERE name = 'u10')
  CREATE USER u10 FOR LOGIN g10;
GO

-- 2. Revoke any direct SELECT on the transactions table
REVOKE SELECT ON dbo.Transaction_Table FROM u10;
GO

-- 3. Create a stored procedure to fetch the 10 most recent receipts
CREATE PROCEDURE dbo.sp_GetRecentReceipts
AS
BEGIN
  SET NOCOUNT ON;
  SELECT TOP 10 *
  FROM dbo.Transaction_Table
  ORDER BY 
    TRY_CONVERT(date, Date_of_Receipt, 101) DESC, 
    Time_of_Receipt DESC;
END;
GO

-- 4. Grant execute on the procedure
GRANT EXECUTE ON dbo.sp_GetRecentReceipts TO u10;
GO

-- 5. Testing section
--    Impersonate the login and user to verify only the proc works

EXECUTE AS LOGIN = 'g10';
  EXECUTE AS USER = 'u10';
  
    -- This should fail with a permission error
    BEGIN TRAN;
      SELECT * FROM dbo.Transaction_Table;
    ROLLBACK;
    
    -- This should succeed and return the top 10 by date/time
    EXEC dbo.sp_GetRecentReceipts;
    
  REVERT;
REVERT;
GO
