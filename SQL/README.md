Project Overview

**EERDBS-3** is a relational database built using Microsoft SQL Server to manage and track retail transactions, purchases, and store locations. It was developed as a **Final Project** to demonstrate advanced database design principles, including programming through triggers and stored procedures, and security mechanisms.

The system serves as a backend for managing Member transactions, providing not only standard data storage but also automated audit logging and analytical reporting capabilities.

--------------------------------------------------------------------------------

Database Schema

The database consists of the following core tables:

-   **Transaction_Table**: The primary entity for receipt headers. It stores critical metadata for every purchase, including `Receipt_No`, `Member_Name`, `Date_of_Receipt`, `Time_of_Receipt`, `Total_Payment`, and `Total_Tax`.
-   **Item_Table**: Stores the individual line items associated with each receipt. It tracks the `Item_Description`, `Quantity`, and `Unit_Price`, linked back to the transaction via `Receipt_No`.
-   **Shop_Table**: A reference table containing physical store information such as `Shop_Name`, `Phone`, and full address details (`Street`, `City`, `State`, `Zip_Code`).
-   **ReceiptsLog** **/** **ItemsLog**: audit tables designed to track changes. They record the `LogDate`, `LoginName`, and `OperationType` (Insert, Update, or Delete) to ensure a complete history of data modifications.

--------------------------------------------------------------------------------

Programmability: Triggers & Stored Procedures

Triggers

The database utilizes triggers to automate data integrity and auditing:

-   **trg10** **(Cleanup Trigger)**: Associated with the `Item_Table`, this trigger fires `AFTER DELETE`. It automatically removes a record from the `Transaction_Table` if the last remaining item for that receipt is deleted, preventing "empty" transaction headers.
-   **trg4** **(Audit Trigger)**: This trigger fires on `INSERT`, `UPDATE`, or `DELETE` operations. It captures session information (using `SYSTEM_USER` and `USER_NAME()`) and logs which receipt was affected into the `ReceiptsLog` table.

Stored Procedures

Standardized operations are handled via precompiled stored procedures:

-   **sp_GetRecentReceipts**: Efficiently retrieves the **10 most recent receipts** from the system, ordered by date and time.
-   **sp2** **(Analytical Reporting)**: This procedure calculates the overall average payment across all transactions and then identifies and displays only those specific receipts where the total payment is **at least 20% higher than the average**.

--------------------------------------------------------------------------------

Security Considerations

Security was a primary focus during development, incorporating both standard and custom measures:

-   **User and Role Management**: The database defines specific users (`u1` through `u10`) and custom roles (e.g., `dbr6`) to enforce the principle of least privilege.
-   **Row-Level Security through Views**: The **vw_MyReceipts** view provides a layer of security by using the `USER_NAME()` function. It ensures that when a member queries their receipts, the system filters the results so they can **only see their own data**, even if they have access to the view.
-   **Execution Contexts**: Stored procedures and triggers were designed to be used with specific `EXECUTE AS` contexts to protect underlying tables from direct user manipulation.

--------------------------------------------------------------------------------
