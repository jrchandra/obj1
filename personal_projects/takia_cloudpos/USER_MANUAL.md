# Hotel Takia CloudPOS — User Manual (Development Release)

## Access
Open the application URL provided by your administrator. Sign in with your staff email and password. Accounts are created by management; the initial admin comes from `.env`.

## Cashier: record a sale
1. Select Bar, Restaurant or Reception.
2. Tap products to add them to the order. The displayed total is VAT-inclusive based on the development tax assumption.
3. Select Cash, EFTPOS or Card. Only record the sale after the external payment is confirmed; this app does not process card payments.
4. Click **Record sale**. The transaction is saved and stock is reduced.
5. Check the Sales tab for the recorded transaction.

## Room charge — test workflow ONLY
Select Charge to room, enter room number and guest name, and acknowledge the warning. The transaction is recorded with `pending_preno` status. **It is NOT posted to Preno.** Hotel staff must not treat this as a confirmed guest charge. This workflow is for testing until the Preno connector is authorised and completed.

## Manager: products and stock
Go to Products & stock, enter product name, category, price and opening stock, then Add product. Use Adjust to record a stock movement and its reason. Avoid using adjustments as a substitute for purchases or refunds.

## Manager: staff
Go to Staff, enter email, a strong password of at least 12 characters and role. Managers can create cashier accounts; only admins can create managers.

## Reporting
Dashboard shows today's sales in UTC, outlet totals, unverified Preno charges and non-fiscalised transactions. Sales shows recent transactions. **Download reconciliation CSV** exports all recorded transactions for accountant review. It is not a ready-to-import MYOB journal.

## Critical operating limitations
No fiscal invoices or receipts are issued. Do not use for FRCS VMS-regulated live sales. Do not rely on the app to validate rooms, post to Preno, send accounting entries to MYOB or take card payments. No refunds/voids, kitchen tickets, shift close, outlet-level inventory or offline mode yet. This is a test/development release.
