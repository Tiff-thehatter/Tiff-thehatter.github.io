USE [EERDBS-3];
GO

CREATE USER u8 WITHOUT LOGIN;
GO

BEGIN TRY
    BEGIN TRY
        EXEC('GRANT EXECUTE ON dbo.sp9 TO u8');
--       PRINT 'Granted EXECUTE on sp9 (if not already granted)';
    END TRY
    BEGIN CATCH
        PRINT 'Note: ' + ERROR_MESSAGE();
    END CATCH
    
    BEGIN TRY
        EXEC('GRANT CREATE TABLE TO u8');
--     PRINT 'Granted CREATE TABLE permission (if not already granted)';
    END TRY
    BEGIN CATCH
        PRINT 'Note: ' + ERROR_MESSAGE();
    END CATCH
END TRY
BEGIN CATCH
    PRINT 'Configuration note: ' + ERROR_MESSAGE();
END CATCH
GO

BEGIN TRY
    PRINT '=== Testing sp9 execution ===';
    
    IF NOT EXISTS (SELECT 1 FROM sys.procedures WHERE name = 'sp9')
    BEGIN
        PRINT 'Error: sp9 procedure not found';
        RETURN;
    END
    
    BEGIN TRY
        EXECUTE AS USER = 'u8';
        BEGIN TRY
            DECLARE @Result MONEY;
            BEGIN TRAN;
            EXEC dbo.sp9 @TotalWeekendExpense = @Result OUTPUT;
            PRINT 'Success! sp9 executed with result: $' + ISNULL(CONVERT(VARCHAR, @Result, 1), '0.00');
            ROLLBACK TRAN;
        END TRY
        BEGIN CATCH
            PRINT 'Execution failed: ' + ERROR_MESSAGE();
            IF @@TRANCOUNT > 0 ROLLBACK TRAN;
        END CATCH
        REVERT;
    END TRY
    BEGIN CATCH
        PRINT 'Impersonation failed: ' + ERROR_MESSAGE();
    END CATCH
    
    PRINT '=== Test complete ===';
END TRY
BEGIN CATCH
    PRINT 'Testing error: ' + ERROR_MESSAGE();
END CATCH
GO
