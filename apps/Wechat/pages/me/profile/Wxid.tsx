import React from 'react';
import { dimens } from '../../../res/dimens';
import { IcMessageSquare } from '../../../res/icons';

export const WxidPage = () => (
    <div className="min-h-full bg-app-surface flex flex-col items-center pt-20">
        <div className="mb-8 opacity-80">
            {/* Using MessageSquare to simulate WeChat Icon */}
            <div className="w-20 h-20 bg-(--app-c-me-avatar-bg) rounded-[16px] flex items-center justify-center">
                <IcMessageSquare size={dimens.icSizePlaceholder} className="text-(--app-c-me-chevron-color)" fill="currentColor" />
            </div>
        </div>
        <div className="text-(--app-title-text-size-18) font-bold text-app-text mb-4">
            WeChat ID: wxid_w5q69z0jbsuj22
        </div>
        <p className="text-(--app-c-tw-text-gray-500) text-(--app-search-filter-text-size) px-12 text-center leading-relaxed">
            WeChat ID is the unique credential for your account and can only be changed once a year.
        </p>
         <div className="mt-auto mb-24 flex justify-center w-full">
            <button className="w-(--app-item-width-184) bg-(--app-c-me-avatar-bg) text-(--app-c-common-text-primary) py-2.5 rounded-[8px] font-bold text-(--app-settings-item-text-size) active:bg-(--app-c-tw-bg-gray-200)">
                Change WeChat ID
            </button>
        </div>
    </div>
);