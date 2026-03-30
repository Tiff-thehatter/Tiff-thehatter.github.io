USE [EERDBS-3];
GO

IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'u6')
    CREATE USER u6 WITHOUT LOGIN;
ELSE
    PRINT 'User u6 already exists, skipping creation.';

IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'u7')
    CREATE USER u7 WITHOUT LOGIN;
ELSE
    PRINT 'User u7 already exists, skipping creation.';
GO

IF NOT EXISTS (
    SELECT 1 
    FROM sys.database_role_members drm
    JOIN sys.database_principals r ON drm.role_principal_id = r.principal_id
    JOIN sys.database_principals m ON drm.member_principal_id = m.principal_id
    WHERE r.name = 'dbr6' AND m.name = 'u6'
)
    ALTER ROLE dbr6 ADD MEMBER u6;
ELSE
    PRINT 'User u6 is already a member of dbr6, skipping.';

IF NOT EXISTS (
    SELECT 1 
    FROM sys.database_role_members drm
    JOIN sys.database_principals r ON drm.role_principal_id = r.principal_id
    JOIN sys.database_principals m ON drm.member_principal_id = m.principal_id
    WHERE r.name = 'dbr6' AND m.name = 'u7'
)
    ALTER ROLE dbr6 ADD MEMBER u7;
ELSE
    PRINT 'User u7 is already a member of dbr6, skipping.';
GO

IF NOT EXISTS (
    SELECT 1 
    FROM sys.database_role_members drm
    JOIN sys.database_principals r ON drm.role_principal_id = r.principal_id
    JOIN sys.database_principals m ON drm.member_principal_id = m.principal_id
    WHERE r.name = 'db_datawriter' AND m.name = 'dbr6'
)
    ALTER ROLE db_datawriter ADD MEMBER dbr6;
ELSE
    PRINT 'Role dbr6 is already a member of db_datawriter, skipping.';
GO

IF EXISTS (SELECT 1 FROM sys.tables WHERE name = 'Transaction_Table')
    DENY UPDATE ON dbo.Transaction_Table(Total_Payment) TO dbr6;
ELSE
    PRINT 'Transaction_Table not found, skipping DENY UPDATE.';
GO

BEGIN TRY
    PRINT '=== Testing u6 permissions ===';
    
    EXECUTE AS USER = 'u6';
    
    BEGIN TRY
        PRINT 'Testing SELECT on Shop_Table...';
        SELECT TOP 1 * FROM dbo.Shop_Table;
        PRINT 'u6 can SELECT from Shop_Table - PASS';
    END TRY
    BEGIN CATCH
        PRINT 'u6 cannot SELECT from Shop_Table - FAIL: ' + ERROR_MESSAGE();
    END CATCH
    
    BEGIN TRY
        PRINT 'Testing SELECT on Transaction_Table...';
        SELECT TOP 1 * FROM dbo.Transaction_Table;
        PRINT 'u6 can SELECT from Transaction_Table - PASS';
    END TRY
    BEGIN CATCH
        PRINT 'u6 cannot SELECT from Transaction_Table - FAIL: ' + ERROR_MESSAGE();
    END CATCH
    
    BEGIN TRY
        PRINT 'Testing EXECUTE on udf7...';
        SELECT * FROM dbo.udf7(10, 100);
        PRINT 'u6 can EXECUTE udf7 - PASS';
    END TRY
    BEGIN CATCH
        PRINT 'u6 cannot EXECUTE udf7 - FAIL: ' + ERROR_MESSAGE();
    END CATCH
    
    BEGIN TRY
        PRINT 'Testing INSERT on Transaction_Table...';
        BEGIN TRAN;
        INSERT INTO dbo.Transaction_Table (
            receipt_no, 
            Member_name, 
            sales_associate, 
            Date_of_Receipt, 
            Time_of_Receipt, 
            Total_Payment, 
            Payment_Method, 
            Total_Tax, 
            Street_Number_PK
        ) VALUES (
            999,                
            'Test Member',      
            'Test Associate',  
            GETDATE(),          
            CONVERT(TIME, GETDATE()), 
            10.00,            
            'Cash',             
            0.50,               
            1                  
        );
        PRINT 'u6 can INSERT into Transaction_Table - PASS';
        ROLLBACK TRAN;
    END TRY
    BEGIN CATCH
        PRINT 'u6 cannot INSERT into Transaction_Table - FAIL: ' + ERROR_MESSAGE();
        IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    END CATCH
 
    BEGIN TRY
        PRINT 'Testing UPDATE on allowed column (Date_of_Receipt)...';
        BEGIN TRAN;
        UPDATE dbo.Transaction_Table 
        SET Date_of_Receipt = GETDATE() 
        WHERE receipt_no = (SELECT TOP 1 receipt_no FROM dbo.Transaction_Table);
        PRINT 'u6 can UPDATE allowed columns - PASS';
        ROLLBACK TRAN;
    END TRY
    BEGIN CATCH
        PRINT 'u6 cannot UPDATE allowed columns - FAIL: ' + ERROR_MESSAGE();
        IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    END CATCH
    
    BEGIN TRY
        PRINT 'Testing UPDATE on denied column (Total_Payment)...';
        BEGIN TRAN;
        UPDATE dbo.Transaction_Table 
        SET Total_Payment = 100 
        WHERE receipt_no = (SELECT TOP 1 receipt_no FROM dbo.Transaction_Table);
        PRINT 'u6 was able to UPDATE Total_Payment - FAIL';
        ROLLBACK TRAN;
    END TRY
    BEGIN CATCH
        PRINT 'u6 cannot UPDATE Total_Payment - PASS: ' + ERROR_MESSAGE();
        IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    END CATCH
    
    REVERT;
    
    PRINT '=== Testing u7 permissions ===';
    
    EXECUTE AS USER = 'u7';
    
    BEGIN TRY
        PRINT 'Testing SELECT on Shop_Table...';
        SELECT TOP 1 * FROM dbo.Shop_Table;
        PRINT 'u7 can SELECT from Shop_Table - PASS';
    END TRY
    BEGIN CATCH
        PRINT 'u7 cannot SELECT from Shop_Table - FAIL: ' + ERROR_MESSAGE();
    END CATCH
    
    BEGIN TRY
        PRINT 'Testing SELECT on Transaction_Table...';
        SELECT TOP 1 * FROM dbo.Transaction_Table;
        PRINT 'u7 can SELECT from Transaction_Table - PASS';
    END TRY
    BEGIN CATCH
        PRINT 'u7 cannot SELECT from Transaction_Table - FAIL: ' + ERROR_MESSAGE();
    END CATCH
    
    BEGIN TRY
        PRINT 'Testing EXECUTE on udf7...';
        SELECT * FROM dbo.udf7(10, 100);
        PRINT 'u7 can EXECUTE udf7 - PASS';
    END TRY
    BEGIN CATCH
        PRINT 'u7 cannot EXECUTE udf7 - FAIL: ' + ERROR_MESSAGE();
    END CATCH
    
    BEGIN TRY
        PRINT 'Testing INSERT on Transaction_Table...';
        BEGIN TRAN;
        INSERT INTO dbo.Transaction_Table (
            receipt_no, 
            Member_name, 
            sales_associate, 
            Date_of_Receipt, 
            Time_of_Receipt, 
            Total_Payment, 
            Payment_Method, 
            Total_Tax, 
            Street_Number_PK
        ) VALUES (
            998,                
            'Test Member 2',   
            'Test Associate 2', 
            GETDATE(),          
            CONVERT(TIME, GETDATE()),
            15.00,             
            'Credit',          
            0.75,               
            2                  
        );
        PRINT 'u7 can INSERT into Transaction_Table - PASS';
        ROLLBACK TRAN;
    END TRY
    BEGIN CATCH
        PRINT 'u7 cannot INSERT into Transaction_Table - FAIL: ' + ERROR_MESSAGE();
        IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    END CATCH
    
    BEGIN TRY
        PRINT 'Testing UPDATE on allowed column (Date_of_Receipt)...';
        BEGIN TRAN;
        UPDATE dbo.Transaction_Table 
        SET Date_of_Receipt = GETDATE() 
        WHERE receipt_no = (SELECT TOP 1 receipt_no FROM dbo.Transaction_Table);
        PRINT 'u7 can UPDATE allowed columns - PASS';
        ROLLBACK TRAN;
    END TRY
    BEGIN CATCH
        PRINT 'u7 cannot UPDATE allowed columns - FAIL: ' + ERROR_MESSAGE();
        IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    END CATCH
    
    BEGIN TRY
        PRINT 'Testing UPDATE on denied column (Total_Payment)...';
        BEGIN TRAN;
        UPDATE dbo.Transaction_Table 
        SET Total_Payment = 100 
        WHERE receipt_no = (SELECT TOP 1 receipt_no FROM dbo.Transaction_Table);
        PRINT 'u7 was able to UPDATE Total_Payment - FAIL';
        ROLLBACK TRAN;
    END TRY
    BEGIN CATCH
        PRINT 'u7 cannot UPDATE Total_Payment - PASS: ' + ERROR_MESSAGE();
        IF @@TRANCOUNT > 0 ROLLBACK TRAN;
    END CATCH
    
    REVERT;
    
    PRINT 'All tests completed successfully.';
END TRY
BEGIN CATCH
    PRINT 'Error occurred during testing: ' + ERROR_MESSAGE();
    
    IF EXISTS (SELECT 1 FROM sys.user_token WHERE principal_id = USER_ID('u6') OR principal_id = USER_ID('u7'))
        REVERT;
    
    IF @@TRANCOUNT > 0
        ROLLBACK;
END CATCH;
GO