Mandatory Task Completion Step: Documentation
Upon completing the technical implementation, you MUST create a documentation file.
Storage Location:
    • All documentation files must be stored in the documentation/ folder at the root of the repository.
    • If the documentation/ directory does not exist, you must create it before saving the file.
File Naming Convention:
[task_name]_[YYYY-MM-DD_HH-mm-ss].info.txt
Required Content Structure:
    1. Summary of Changes: Provide a high-level overview of the integration, such as new model support (e.g., mapping "dolphin-24B" to "dphn/Dolphin3.0-R1-Mistral-24B")22222.
    2. Detailed Implementation Notes:
        ◦ List specific files created or updated, such as implementing custom classes in src/ollm/mistral.py to handle layer-wise offloading3.
        ◦ Explain architectural decisions, like why a dedicated module was used instead of an existing one to ensure correct monkey-patching4.
    3. Fixes & Caveats:
        ◦ Document hardware constraints, such as the current lack of support for multi-GPU lists or use_kvikio parameters5555.
        ◦ Explain bug fixes identified, such as resolving NoneType errors related to the loader manifest or fixing missing attention_mask warnings6666.
    4. Usage Instructions:
        ◦ Provide the exact command to run new scripts, including necessary environment variables like PYTHONPATH=src7.
        ◦ List any new demonstration scripts created (e.g., scripts/example_dolphin_mistral.py)8.
    5. Technical Metrics: Include specific configurations used for testing, such as temperature=0.1, do_sample=True, and default offloading layer counts9.

Key Changes Made:
    • Directory Shift: The instruction now explicitly points to the documentation/ folder instead of the scripts/ directory previously used10.
    • Auto-Creation Logic: Added a specific instruction for Jules to check for and create the documentation/ folder if it is missing, ensuring your repository structure remains organized automatically.
    • Schema Adherence: Maintained the "timestamped info file" requirement and Q&A/Fixes format seen in your previous successful tasks111111111111111111.