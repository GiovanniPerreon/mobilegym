import React, { useState, useEffect, useRef } from 'react';
import { dimens } from '../../../res/dimens';
import { useWechatStrings } from '../../../hooks/useWechatStrings';
import { IcNavForward, IcLocation, IcScan } from '../../../res/icons';
import { useWechatStore } from '../../../state';
import { useShallow } from 'zustand/react/shallow';
import { Address } from '../../../types';
import * as TimeService from '../../../../../os/TimeService';
import { useWechatGestures } from '../../../hooks/useWechatGestures';

let localSeq = 0;
function nextAddressId(): string {
    localSeq += 1;
    return `address_${TimeService.now()}_${localSeq}`;
}

export const AddressListPage = () => {
    const addresses = useWechatStore(s => s.user.addresses);

    if (addresses.length === 0) {
        return (
            <div className="min-h-full bg-app-surface flex flex-col items-center justify-center -mt-20">
                <div className="text-(--app-c-tw-text-gray-400) text-(--app-chat-bubble-text-size) mb-2">No address information yet</div>
                <div className="text-(--app-c-address-link-text) text-(--app-chat-bubble-text-size)">Add Address</div>
            </div>
        );
    }

    return (
        <div className="min-h-full bg-app-bg">
            {addresses.map((addr, idx) => (
                <div key={idx} className="bg-app-surface p-4 mb-2 flex justify-between items-center">
                    <div>
                        <div className="text-app-text text-(--app-chat-bubble-text-size) font-medium mb-1">{addr.name}, {addr.phone}</div>
                        <div className="text-(--app-c-tw-text-gray-500) text-sm">{addr.region} {addr.detail}</div>
                    </div>
                </div>
            ))}
        </div>
    );
};

export const AddAddressPage = () => {
    const t = useWechatStrings();
    const { addAddress, setRightAction } = useWechatStore(useShallow(s => ({
        addAddress: s.addAddress,
        setRightAction: s.setRightAction,
    })));
    const { back } = useWechatGestures();
    const [form, setForm] = useState<Address>({
        id: nextAddressId(),
        name: '',
        phone: '',
        region: '',
        detail: ''
    });
    const formRef = useRef(form);

    useEffect(() => {
        formRef.current = form;
    }, [form]);

    const handleChange = (field: keyof Address, value: string) => {
        setForm(prev => ({ ...prev, [field]: value }));
    };

    useEffect(() => {
        setRightAction({
            onTrigger: () => {
            const currentForm = formRef.current;
            if (!currentForm.name || !currentForm.phone) return; // Basic validation
            addAddress(currentForm);
            back();
            },
        });
        return () => setRightAction(null);
    }, [addAddress, setRightAction, back]);

    // Helper to render each field row
    const Field = ({ label, placeholder, field }: { label: string, placeholder: string, field: keyof Address }) => {
        // Show right arrow for region field
        const showArrow = field === 'region';
        // Show country code + arrow for phone field
        const showPhoneCode = field === 'phone';
        return (
            <div className="flex items-center py-4 border-b border-(--app-c-tw-border-gray-100)">
                <span className="w-24 text-(--app-chat-bubble-text-size) text-app-text">{label}</span>
                <input
                    type="text"
                    placeholder={placeholder}
                    value={form[field]}
                    onChange={(e) => handleChange(field, e.target.value)}
                    className="flex-1 outline-none text-(--app-chat-bubble-text-size)"
                />
                {showArrow && <IcNavForward size={dimens.icSizeChevronSm} className="text-(--app-c-tw-text-gray-300)" />}
                {showPhoneCode && <span className="text-(--app-chat-bubble-text-size) text-app-text mr-2">+86</span>}
                {showPhoneCode && <IcNavForward size={dimens.icSizeChevronSm} className="text-(--app-c-tw-text-gray-300)" />}
            </div>
        );
    };

    return (
        <div className="min-h-full bg-app-surface px-4">
            <div className="flex text-sm text-(--app-c-address-link-text) py-4 mb-2">
                <span className="flex items-center mr-6"><IcLocation size={dimens.icSizeTiny} className="mr-1" /> Select from map</span>
                <span className="flex items-center"><IcScan size={dimens.icSizeTiny} className="mr-1" /> Recognize from clipboard</span>
            </div>

            <Field label="Region" placeholder="Select province, city, district and street" field="region" />
            <Field label="Detailed Address" placeholder="Enter detailed location and house number" field="detail" />
            <Field label="Full Name" placeholder="Enter your name" field="name" />
            <Field label={t.contacts_phone} placeholder="Enter phone number" field="phone" />
        </div>
    );
};