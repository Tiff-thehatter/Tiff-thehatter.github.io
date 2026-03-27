CREATE OR ALTER FUNCTION dbo.udf10 (@Rank INT)
RETURNS VARCHAR(20)
AS
BEGIN
    DECLARE @ZipCode VARCHAR(20);

    SELECT @ZipCode = v.VendorZipCode
    FROM (
        SELECT
            v.VendorID,
            v.VendorZipCode,
            DENSE_RANK() OVER (ORDER BY COUNT(i.InvoiceNumber) DESC) AS ReceiptRank
        FROM
            Vendors v
        LEFT JOIN
            Invoices i ON v.VendorID = i.VendorID
        GROUP BY
            v.VendorID, v.VendorZipCode
    ) AS RankedVendors
    WHERE
        ReceiptRank = @Rank;

    IF @ZipCode IS NULL
    BEGIN
        RETURN '99999';
    END

    RETURN @ZipCode;
END;
GO

-- Testing script
SELECT 'Zipcode for rank 1: ' + dbo.udf10(1);
SELECT 'Zipcode for rank 3: ' + dbo.udf10(3);
SELECT 'Zipcode for rank 100 (unlikely to exist): ' + dbo.udf10(100);