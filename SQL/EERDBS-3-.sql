USE [master]
GO
/****** Object:  Database [EERDBS-3]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE DATABASE [EERDBS-3]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'EERDBS-3', FILENAME = N'C:\Program Files\Microsoft SQL Server\EERBS-3--\EERDBS-3.mdf' , SIZE = 8192KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'EERDBS-3_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\EERBS-3--\EERDBS-3_log.ldf' , SIZE = 8192KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [EERDBS-3] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [EERDBS-3].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [EERDBS-3] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [EERDBS-3] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [EERDBS-3] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [EERDBS-3] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [EERDBS-3] SET ARITHABORT OFF 
GO
ALTER DATABASE [EERDBS-3] SET AUTO_CLOSE ON 
GO
ALTER DATABASE [EERDBS-3] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [EERDBS-3] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [EERDBS-3] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [EERDBS-3] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [EERDBS-3] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [EERDBS-3] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [EERDBS-3] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [EERDBS-3] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [EERDBS-3] SET  DISABLE_BROKER 
GO
ALTER DATABASE [EERDBS-3] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [EERDBS-3] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [EERDBS-3] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [EERDBS-3] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [EERDBS-3] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [EERDBS-3] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [EERDBS-3] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [EERDBS-3] SET RECOVERY SIMPLE 
GO
ALTER DATABASE [EERDBS-3] SET  MULTI_USER 
GO
ALTER DATABASE [EERDBS-3] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [EERDBS-3] SET DB_CHAINING OFF 
GO
ALTER DATABASE [EERDBS-3] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [EERDBS-3] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [EERDBS-3] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [EERDBS-3] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
ALTER DATABASE [EERDBS-3] SET QUERY_STORE = ON
GO
ALTER DATABASE [EERDBS-3] SET QUERY_STORE (OPERATION_MODE = READ_WRITE, CLEANUP_POLICY = (STALE_QUERY_THRESHOLD_DAYS = 30), DATA_FLUSH_INTERVAL_SECONDS = 900, INTERVAL_LENGTH_MINUTES = 60, MAX_STORAGE_SIZE_MB = 1000, QUERY_CAPTURE_MODE = AUTO, SIZE_BASED_CLEANUP_MODE = AUTO, MAX_PLANS_PER_QUERY = 200, WAIT_STATS_CAPTURE_MODE = ON)
GO
USE [EERDBS-3]
GO
/****** Object:  User [u5]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE USER [u5] FOR LOGIN [g5] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [u4]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE USER [u4] FOR LOGIN [g4] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [u2]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE USER [u2] FOR LOGIN [g2] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [u10]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE USER [u10] FOR LOGIN [g10] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [u1]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE USER [u1] FOR LOGIN [g1] WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [Tiffany]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE USER [Tiffany] WITHOUT LOGIN WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [Micaela]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE USER [Micaela] WITHOUT LOGIN WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [Jordan]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE USER [Jordan] WITHOUT LOGIN WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [Ismael]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE USER [Ismael] WITHOUT LOGIN WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  User [Alena]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE USER [Alena] WITHOUT LOGIN WITH DEFAULT_SCHEMA=[dbo]
GO
/****** Object:  DatabaseRole [dbr6]    Script Date: 4/22/2025 8:51:29 PM ******/
CREATE ROLE [dbr6]
GO
ALTER ROLE [db_owner] ADD MEMBER [u1]
GO
/****** Object:  UserDefinedFunction [dbo].[udf1]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

----------------------------------------------------
-- 1) CREATE OR ALTER FUNCTION: udf1
--    Returns the total expense of all receipts
----------------------------------------------------
CREATE   FUNCTION [dbo].[udf1]()
RETURNS MONEY
AS
BEGIN
    DECLARE @TotalExpense MONEY;

    SELECT @TotalExpense = SUM(Total_Payment)
    FROM dbo.Transaction_Table;

    RETURN ISNULL(@TotalExpense, 0);
END;
GO
/****** Object:  UserDefinedFunction [dbo].[udf2]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

------------------------------------------------------------
-- 1) CREATE OR ALTER FUNCTION: udf2
--    Returns a table (Month, Year, Average Expense)
------------------------------------------------------------
CREATE   FUNCTION [dbo].[udf2]()
RETURNS @MonthlyAverages TABLE 
(
    [Month]           NVARCHAR(20),
    [Year]            INT,
    [Average Expense] MONEY
)
AS
BEGIN
    INSERT INTO @MonthlyAverages
    SELECT 
       DATENAME(MONTH, TRY_CAST(Date_of_Receipt AS DATE)) AS [Month],
       YEAR(TRY_CAST(Date_of_Receipt AS DATE))            AS [Year],
       AVG(Total_Payment)                                 AS [Average Expense]
    FROM dbo.Transaction_Table
    -- Exclude rows where the date cannot be parsed
    WHERE TRY_CAST(Date_of_Receipt AS DATE) IS NOT NULL
    GROUP BY 
       DATENAME(MONTH, TRY_CAST(Date_of_Receipt AS DATE)),
       YEAR(TRY_CAST(Date_of_Receipt AS DATE));

    RETURN;
END;
GO
/****** Object:  Table [dbo].[Transaction_Table]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Transaction_Table](
	[Receipt_No] [tinyint] NOT NULL,
	[Member_Name] [nvarchar](50) NOT NULL,
	[sales_associate] [nvarchar](50) NULL,
	[Date_of_Receipt] [nvarchar](50) NOT NULL,
	[Time_of_Receipt] [time](7) NOT NULL,
	[Total_Payment] [money] NOT NULL,
	[Payment_Method] [nvarchar](50) NULL,
	[Total_Tax] [money] NOT NULL,
	[Street_Number_PK] [smallint] NULL,
 CONSTRAINT [PK_Transaction_Table] PRIMARY KEY CLUSTERED 
(
	[Receipt_No] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[vw_MyReceipts]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

-- 2. Create (or replace) the view that only returns the current user's receipts
CREATE   VIEW [dbo].[vw_MyReceipts]
AS
  SELECT *
  FROM dbo.Transaction_Table
  WHERE Member_Name = USER_NAME();
GO
/****** Object:  Table [dbo].[Item_Table]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Item_Table](
	[I_D] [smallint] NOT NULL,
	[Quantity] [float] NULL,
	[Item_Description] [nvarchar](50) NOT NULL,
	[Unit_Price] [money] NULL,
	[Receipt_No] [tinyint] NULL,
 CONSTRAINT [PK_Item_Table] PRIMARY KEY CLUSTERED 
(
	[I_D] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  UserDefinedFunction [dbo].[udf7]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE   FUNCTION [dbo].[udf7] (
    @MinUnitPrice money,
    @MaxUnitPrice money
)
RETURNS TABLE
AS
RETURN (
    SELECT
        it.Receipt_No AS recieptid,
        it.Item_Description AS itemdescription,
        it.Unit_Price AS price,
        it.I_D AS ItemID
    FROM
        Item_Table it
    WHERE
        it.Unit_Price BETWEEN @MinUnitPrice AND @MaxUnitPrice
);
GO
/****** Object:  Table [dbo].[ItemLog]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ItemLog](
	[LogDate] [datetime] NOT NULL,
	[LoginName] [varchar](128) NULL,
	[UserName] [varchar](128) NULL,
	[OperationType] [char](1) NOT NULL,
	[ItemAffected] [smallint] NULL,
	[OldPrice] [money] NULL,
	[NewPrice] [money] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[ItemsLog]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[ItemsLog](
	[LogDate] [datetime] NOT NULL,
	[LoginName] [varchar](128) NULL,
	[UserName] [varchar](128) NULL,
	[OperationType] [char](1) NOT NULL,
	[ItemAffected] [smallint] NULL,
	[OldPrice] [money] NULL,
	[NewPrice] [money] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[Shop_Table]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Shop_Table](
	[Shop_Name] [nvarchar](100) NOT NULL,
	[Phone] [bigint] NULL,
	[Street_Number_PK] [smallint] NOT NULL,
	[Street] [nvarchar](50) NOT NULL,
	[City] [nvarchar](50) NULL,
	[State] [nvarchar](50) NOT NULL,
	[Zip_Code] [int] NULL,
	[Website] [nvarchar](50) NULL,
 CONSTRAINT [PK_Shop_Table] PRIMARY KEY CLUSTERED 
(
	[Street_Number_PK] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
ALTER TABLE [dbo].[Item_Table]  WITH NOCHECK ADD  CONSTRAINT [FK_Item_Table_Transaction_Table] FOREIGN KEY([Receipt_No])
REFERENCES [dbo].[Transaction_Table] ([Receipt_No])
GO
ALTER TABLE [dbo].[Item_Table] CHECK CONSTRAINT [FK_Item_Table_Transaction_Table]
GO
ALTER TABLE [dbo].[Transaction_Table]  WITH CHECK ADD  CONSTRAINT [FK_Transaction_Table_Shop_Table] FOREIGN KEY([Street_Number_PK])
REFERENCES [dbo].[Shop_Table] ([Street_Number_PK])
GO
ALTER TABLE [dbo].[Transaction_Table] CHECK CONSTRAINT [FK_Transaction_Table_Shop_Table]
GO
ALTER TABLE [dbo].[Transaction_Table]  WITH CHECK ADD  CONSTRAINT [FK_Transaction_Table_Transaction_Table] FOREIGN KEY([Receipt_No])
REFERENCES [dbo].[Transaction_Table] ([Receipt_No])
GO
ALTER TABLE [dbo].[Transaction_Table] CHECK CONSTRAINT [FK_Transaction_Table_Transaction_Table]
GO
/****** Object:  StoredProcedure [dbo].[sp_GetRecentReceipts]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

-- 3. Create a stored procedure to fetch the 10 most recent receipts
CREATE PROCEDURE [dbo].[sp_GetRecentReceipts]
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
/****** Object:  StoredProcedure [dbo].[sp2]    Script Date: 4/22/2025 8:51:29 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

------------------------------------------------------------
-- 1) CREATE OR ALTER PROCEDURE: sp2
--    Output Parameters:
--        @ReceiptCount   INT
--        @AveragePayment MONEY
--    Functionality:
--      - Calculate number of receipts
--      - Calculate overall average total amount
--      - Display receipts >= 120% of that average
------------------------------------------------------------
CREATE   PROCEDURE [dbo].[sp2]
    @ReceiptCount   INT    OUTPUT,
    @AveragePayment MONEY  OUTPUT
AS
BEGIN
    SET NOCOUNT ON;

    ---------------------------------------------------------
    -- Step A: Calculate total receipt count and average amount
    ---------------------------------------------------------
    SELECT 
        @ReceiptCount   = COUNT(*),
        @AveragePayment = AVG(Total_Payment)
    FROM dbo.Transaction_Table;

    ---------------------------------------------------------
    -- Step B: Display receipts whose total is at least
    --         20% higher than overall average
    ---------------------------------------------------------
    SELECT 
        Receipt_No     AS [Receipt ID],
        Total_Payment  AS [Total Amount]
    FROM dbo.Transaction_Table
    WHERE Total_Payment >= 1.2 * @AveragePayment;

END;
GO
USE [master]
GO
ALTER DATABASE [EERDBS-3] SET  READ_WRITE 
GO
