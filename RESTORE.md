# Emergency Data Restore

If database is corrupted or deleted:

1. Stop the server
2. Find latest backup:
```
   ls -la backups/
```

3. Restore:
```
   cp backups/backup_YYYYMMDD_HHMMSS/milvus_demo.db ./
   cp -r backups/backup_YYYYMMDD_HHMMSS/images/* data/images/
   cp backups/backup_YYYYMMDD_HHMMSS/claims.json data/
```

4. Restart server
5. Verify stats show correct pet count