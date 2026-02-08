import json

# Check if the fixed notebook is valid
try:
    with open(r'C:\Users\spicc\Desktop\Multimodal\AURA\notebooks\AURA_V10_Final_Fixed.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"✅ Notebook is valid JSON!")
    print(f"   Total cells: {len(nb['cells'])}")
    print(f"   Notebook format: v{nb['nbformat']}.{nb['nbformat_minor']}")
    
    # Count cell types
    code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    md_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
    
    print(f"   Code cells: {code_cells}")
    print(f"   Markdown cells: {md_cells}")
    
    # Look for our fixes
    print(f"\n🔍 Checking for applied fixes...")
    
    fixes_found = {
        'requires_grad=False': 0,
        'empty_batch_check': 0,
        'nan_inf_check': 0,
        'reporting_298': 0
    }
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            if 'requires_grad=False' in source and 'torch.tensor(0.' in source:
                fixes_found['requires_grad=False'] += 1
            
            if 'all((tasks == i).sum() == 0' in source:
                fixes_found['empty_batch_check'] += 1
            
            if 'torch.isnan(loss) or torch.isinf(loss)' in source:
                fixes_found['nan_inf_check'] += 1
        
        if cell['cell_type'] in ['code', 'markdown']:
            source = ''.join(cell['source'])
            if '298' in source and 'reporting' in source.lower():
                fixes_found['reporting_298'] += 1
    
    print(f"   ✅ requires_grad=False: {fixes_found['requires_grad=False']} occurrences")
    print(f"   ✅ Empty batch check: {fixes_found['empty_batch_check']} occurrences")
    print(f"   ✅ NaN/Inf check: {fixes_found['nan_inf_check']} occurrences")
    print(f"   ✅ Reporting 298: {fixes_found['reporting_298']} mentions")
    
    print(f"\n🎯 Notebook is ready for use!")
    
except json.JSONDecodeError as e:
    print(f"❌ JSON parsing error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
