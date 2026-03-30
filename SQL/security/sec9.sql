USE [EERDBS-3];
GO

-- 1. Create a user (no login) for each receipt owner
CREATE USER [Alena]       WITHOUT LOGIN;
CREATE USER [Ismael]      WITHOUT LOGIN;
CREATE USER [Jordan]      WITHOUT LOGIN;
CREATE USER [Micaela]     WITHOUT LOGIN;
CREATE USER [Tiffany]     WITHOUT LOGIN;
GO

-- 2. Create (or replace) the view that only returns the current user's receipts
CREATE OR ALTER VIEW dbo.vw_MyReceipts
AS
  SELECT *
  FROM dbo.Transaction_Table
  WHERE Member_Name = USER_NAME();
GO

-- 3. Grant each user SELECT on the view, and revoke direct SELECT on the base table
GRANT SELECT ON dbo.vw_MyReceipts       TO [Alena];
REVOKE SELECT ON dbo.Transaction_Table  FROM [Alena];

GRANT SELECT ON dbo.vw_MyReceipts       TO [Ismael];
REVOKE SELECT ON dbo.Transaction_Table  FROM [Ismael];

GRANT SELECT ON dbo.vw_MyReceipts       TO [Jordan];
REVOKE SELECT ON dbo.Transaction_Table  FROM [Jordan];

GRANT SELECT ON dbo.vw_MyReceipts       TO [Micaela];
REVOKE SELECT ON dbo.Transaction_Table  FROM [Micaela];

GRANT SELECT ON dbo.vw_MyReceipts       TO [Tiffany];
REVOKE SELECT ON dbo.Transaction_Table  FROM [Tiffany];
GO

-- 4. Triggers to block UPDATE/DELETE on others’ receipts

-- UPDATE trigger
CREATE OR ALTER TRIGGER trg_Transaction_Update
ON dbo.Transaction_Table
AFTER UPDATE
AS
BEGIN
  IF EXISTS (
    SELECT 1
    FROM inserted i
    WHERE i.Member_Name <> USER_NAME()
  )
  BEGIN
    RAISERROR('Security violation: You cannot update receipts that are not yours.', 16, 1);
    ROLLBACK TRANSACTION;
  END
END;
GO

-- DELETE trigger
CREATE OR ALTER TRIGGER trg_Transaction_Delete
ON dbo.Transaction_Table
INSTEAD OF DELETE
AS
BEGIN
  IF EXISTS (
    SELECT 1
    FROM deleted d
    WHERE d.Member_Name <> USER_NAME()
  )
  BEGIN
    RAISERROR('Security violation: You cannot delete receipts that are not yours.', 16, 1);
    RETURN;
  END

  DELETE T
  FROM dbo.Transaction_Table AS T
  JOIN deleted AS D
    ON T.Receipt_No = D.Receipt_No;
END;
GO

-- 5. Grant UPDATE and DELETE so the triggers can fire
GRANT UPDATE ON dbo.Transaction_Table TO [Alena];
GRANT DELETE ON dbo.Transaction_Table TO [Alena];

GRANT UPDATE ON dbo.Transaction_Table TO [Ismael];
GRANT DELETE ON dbo.Transaction_Table TO [Ismael];

GRANT UPDATE ON dbo.Transaction_Table TO [Jordan];
GRANT DELETE ON dbo.Transaction_Table TO [Jordan];

GRANT UPDATE ON dbo.Transaction_Table TO [Micaela];
GRANT DELETE ON dbo.Transaction_Table TO [Micaela];

GRANT UPDATE ON dbo.Transaction_Table TO [Tiffany];
GRANT DELETE ON dbo.Transaction_Table TO [Tiffany];
GO

-- 6. Testing: impersonate each user (use actual Receipt_No values for your data)

-- Test Alena
EXECUTE AS USER = 'Alena';
  SELECT * FROM dbo.vw_MyReceipts;
  BEGIN TRAN;
    -- assume Receipt_No 101 belongs to Alena
    UPDATE dbo.Transaction_Table SET Total_Payment = Total_Payment WHERE Receipt_No = 101;
  ROLLBACK;
  BEGIN TRAN;
    -- assume Receipt_No 102 belongs to someone else
    DELETE FROM dbo.Transaction_Table WHERE Receipt_No = 102;
  ROLLBACK;
REVERT;
GO

-- Test Ismael
EXECUTE AS USER = 'Ismael';
  SELECT * FROM dbo.vw_MyReceipts;
  BEGIN TRAN;
    UPDATE dbo.Transaction_Table SET Total_Payment = Total_Payment WHERE Receipt_No = 201;
  ROLLBACK;
  BEGIN TRAN;
    DELETE FROM dbo.Transaction_Table WHERE Receipt_No = 202;
  ROLLBACK;
REVERT;
GO

-- Test Jordan
EXECUTE AS USER = 'Jordan';
  SELECT * FROM dbo.vw_MyReceipts;
  BEGIN TRAN;
    UPDATE dbo.Transaction_Table SET Total_Payment = Total_Payment WHERE Receipt_No = 301;
  ROLLBACK;
  BEGIN TRAN;
    DELETE FROM dbo.Transaction_Table WHERE Receipt_No = 302;
  ROLLBACK;
REVERT;
GO

-- Test Micaela
EXECUTE AS USER = 'Micaela';
  SELECT * FROM dbo.vw_MyReceipts;
  BEGIN TRAN;
    UPDATE dbo.Transaction_Table SET Total_Payment = Total_Payment WHERE Receipt_No = 401;
  ROLLBACK;
  BEGIN TRAN;
    DELETE FROM dbo.Transaction_Table WHERE Receipt_No = 402;
  ROLLBACK;
REVERT;
GO

-- Test Tiffany
EXECUTE AS USER = 'Tiffany';
  SELECT * FROM dbo.vw_MyReceipts;
  BEGIN TRAN;
    UPDATE dbo.Transaction_Table SET Total_Payment = Total_Payment WHERE Receipt_No = 501;
  ROLLBACK;
  BEGIN TRAN;
    DELETE FROM dbo.Transaction_Table WHERE Receipt_No = 502;
  ROLLBACK;
REVERT;
GO
