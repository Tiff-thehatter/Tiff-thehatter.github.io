-- sec6.sql

USE [EERDBS-3];
GO

-- Create a user-defined database role named dbr6

CREATE  ROLE dbr6;
GO

-- Grant permissions to the dbr6 role based on the permissions granted to u2 and u3.

GRANT SELECT ON dbo.Shop_Table TO dbr6;
GRANT SELECT ON dbo.Transaction_Table TO dbr6;
GRANT EXECUTE ON dbo.udf7 TO dbr6; -- EXECUTE permission is for executable objects like procedures or functions [7]
GRANT EXECUTE ON dbo.udf1 TO dbr6;
GRANT EXECUTE ON dbo.udf2 TO dbr6;
GRANT INSERT ON dbo.Transaction_Table TO dbr6;
GRANT UPDATE ON dbo.Transaction_Table (Date_of_Receipt) TO dbr6;


GO