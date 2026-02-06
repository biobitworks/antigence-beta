#!/bin/bash
# Create Virtual Machine for IMMUNOS-MCP Testing
# Usage: ./create_vm.sh

set -e

VM_NAME="immunos-mcp-test"
VM_RAM="4096"  # 4GB
VM_DISK="20480"  # 20GB
VM_ISO="debian-12-netinst.iso"

echo "Creating VM for IMMUNOS-MCP testing..."

# Check if VirtualBox is installed
if ! command -v VBoxManage &> /dev/null; then
    echo "Error: VirtualBox not found. Please install VirtualBox first."
    exit 1
fi

# Create VM
echo "Creating VM: $VM_NAME"
VBoxManage createvm --name "$VM_NAME" --ostype "Debian_64" --register

# Configure VM
echo "Configuring VM..."
VBoxManage modifyvm "$VM_NAME" \
    --memory "$VM_RAM" \
    --cpus 2 \
    --vram 16 \
    --graphicscontroller vboxsvga \
    --audio none \
    --usb off \
    --clipboard disabled \
    --draganddrop disabled

# Create disk
echo "Creating virtual disk..."
VBoxManage createhd \
    --filename "$HOME/VirtualBox VMs/$VM_NAME/$VM_NAME.vdi" \
    --size "$VM_DISK" \
    --format VDI

# Attach disk
VBoxManage storagectl "$VM_NAME" \
    --name "SATA Controller" \
    --add sata \
    --controller IntelAHCI

VBoxManage storageattach "$VM_NAME" \
    --storagectl "SATA Controller" \
    --port 0 \
    --device 0 \
    --type hdd \
    --medium "$HOME/VirtualBox VMs/$VM_NAME/$VM_NAME.vdi"

# Attach ISO (if provided)
if [ -f "$VM_ISO" ]; then
    echo "Attaching ISO: $VM_ISO"
    VBoxManage storagectl "$VM_NAME" \
        --name "IDE Controller" \
        --add ide \
        --controller PIIX4
    
    VBoxManage storageattach "$VM_NAME" \
        --storagectl "IDE Controller" \
        --port 0 \
        --device 0 \
        --type dvddrive \
        --medium "$VM_ISO"
else
    echo "Warning: ISO not found. Please attach ISO manually:"
    echo "  VBoxManage storageattach $VM_NAME --storagectl 'IDE Controller' --port 0 --device 0 --type dvddrive --medium /path/to/debian.iso"
fi

# Configure network (NAT for initial setup, can change to Internal Network later)
VBoxManage modifyvm "$VM_NAME" \
    --nic1 nat \
    --natpf1 "guestssh,tcp,,2222,,22"

echo ""
echo "✅ VM created successfully!"
echo ""
echo "Next steps:"
echo "1. Start VM: VBoxManage startvm $VM_NAME --type gui"
echo "2. Install Debian 12 (minimal)"
echo "3. Install IMMUNOS-MCP in VM"
echo "4. Test offline mode"
echo ""
echo "To change network to internal (air-gapped):"
echo "  VBoxManage modifyvm $VM_NAME --nic1 intnet --intnet1 immunos-net"

