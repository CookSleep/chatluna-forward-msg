import { Schema, type Context } from 'koishi';
import { ModelCapabilities, ModelType } from 'koishi-plugin-chatluna/llm-core/platform/types';
export declare const name = "chatluna-forward-msg";
export declare const inject: string[];
type PluginConfig = {
    [key: string]: any;
    protocolService?: Record<string, any>;
    readTool?: Record<string, any>;
    sendTool?: Record<string, any>;
    fakeTool?: Record<string, any>;
    describeImageTool?: Record<string, any>;
    imageService?: Record<string, any>;
    cacheService?: Record<string, any>;
};
type ModelCapabilityValue = typeof ModelCapabilities[keyof typeof ModelCapabilities];
type ChatModel = {
    modelInfo?: {
        capabilities?: ModelCapabilityValue[];
    };
    invoke: (messages: unknown[]) => Promise<{
        content?: unknown;
    }>;
};
type ChatModelInfoLite = {
    platform?: string;
    name?: string;
    capabilities?: ModelCapabilityValue[];
};
type ModelTypeValue = typeof ModelType[keyof typeof ModelType];
type ImageModelRef = {
    value?: ChatModel;
};
type PluginContext = Context & {
    chatluna: {
        createChatModel: (name: string) => Promise<ImageModelRef>;
        platform: {
            listAllModels: (type: ModelTypeValue) => {
                value?: ChatModelInfoLite[];
            };
        };
    };
};
type CacheRow = {
    key: string;
    payload: any;
    createdAt?: string | number | Date;
    expiresAt: string | number | Date;
};
declare module 'koishi' {
    interface Tables {
        chatluna_forward_msg_cache: CacheRow;
    }
}
export declare function apply(ctx: PluginContext, config: PluginConfig): void;
export declare const Config: Schema<Schemastery.ObjectS<{
    protocolService: Schema<Schemastery.ObjectS<{
        enableNapcat: Schema<boolean, boolean>;
        enableLLBot: Schema<boolean, boolean>;
    }>, Schemastery.ObjectT<{
        enableNapcat: Schema<boolean, boolean>;
        enableLLBot: Schema<boolean, boolean>;
    }>>;
    readTool: Schema<Schemastery.ObjectS<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
        maxParseDepth: Schema<number, number>;
        describeImageInRead: Schema<boolean, boolean>;
    }>, Schemastery.ObjectT<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
        maxParseDepth: Schema<number, number>;
        describeImageInRead: Schema<boolean, boolean>;
    }>>;
    sendTool: Schema<Schemastery.ObjectS<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
        botDisplayName: Schema<string, string>;
    }>, Schemastery.ObjectT<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
        botDisplayName: Schema<string, string>;
    }>>;
    fakeTool: Schema<Schemastery.ObjectS<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
    }>, Schemastery.ObjectT<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
    }>>;
    describeImageTool: Schema<Schemastery.ObjectS<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
    }>, Schemastery.ObjectT<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
    }>>;
    imageService: Schema<Schemastery.ObjectS<{
        model: Schema<any, any>;
        prompt: Schema<string, string>;
        taskConcurrency: Schema<number, number>;
        requestTimeoutSeconds: Schema<number, number>;
    }>, Schemastery.ObjectT<{
        model: Schema<any, any>;
        prompt: Schema<string, string>;
        taskConcurrency: Schema<number, number>;
        requestTimeoutSeconds: Schema<number, number>;
    }>>;
    cacheService: Schema<Schemastery.ObjectS<{
        enable: Schema<boolean, boolean>;
        ttlSeconds: Schema<number, number>;
        storagePath: Schema<string, string>;
        cleanupIntervalSeconds: Schema<number, number>;
    }>, Schemastery.ObjectT<{
        enable: Schema<boolean, boolean>;
        ttlSeconds: Schema<number, number>;
        storagePath: Schema<string, string>;
        cleanupIntervalSeconds: Schema<number, number>;
    }>>;
}>, {
    protocolService: Schemastery.ObjectT<{
        enableNapcat: Schema<boolean, boolean>;
        enableLLBot: Schema<boolean, boolean>;
    }>;
    readTool: Schemastery.ObjectT<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
        maxParseDepth: Schema<number, number>;
        describeImageInRead: Schema<boolean, boolean>;
    }>;
    sendTool: Schemastery.ObjectT<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
        botDisplayName: Schema<string, string>;
    }>;
    fakeTool: Schemastery.ObjectT<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
    }>;
    describeImageTool: Schemastery.ObjectT<{
        enable: Schema<boolean, boolean>;
        name: Schema<string, string>;
        description: Schema<string, string>;
    }>;
    imageService: Schemastery.ObjectT<{
        model: Schema<any, any>;
        prompt: Schema<string, string>;
        taskConcurrency: Schema<number, number>;
        requestTimeoutSeconds: Schema<number, number>;
    }>;
    cacheService: Schemastery.ObjectT<{
        enable: Schema<boolean, boolean>;
        ttlSeconds: Schema<number, number>;
        storagePath: Schema<string, string>;
        cleanupIntervalSeconds: Schema<number, number>;
    }>;
} & import("cosmokit").Dict>;
export declare const usage = "\n## chatluna-forward-msg\n\n**\u4F7F\u7528\u672C\u63D2\u4EF6\u5373\u4EE3\u8868\u60A8\u5DF2\u9605\u8BFB\u5E76\u540C\u610F\u4EE5\u4E0B\u5185\u5BB9\uFF0C\u5982\u4E0D\u540C\u610F\uFF0C\u8BF7\u7ACB\u5373\u5378\u8F7D\u672C\u63D2\u4EF6\uFF1A**\n- \u672C\u63D2\u4EF6\u4E2D\u7684\u4F2A\u9020\u8EAB\u4EFD\u53D1\u9001\u5408\u5E76\u8F6C\u53D1\u80FD\u529B\u4EC5\u7528\u4E8E\u5F00\u53D1\u6D4B\u8BD5\u4E0E\u5A31\u4E50\u573A\u666F\u3002\n- \u4F7F\u7528\u8005\u9700\u81EA\u884C\u786E\u4FDD\u7B26\u5408\u5F53\u5730\u6CD5\u5F8B\u6CD5\u89C4\u4E0E\u5E73\u53F0\u89C4\u5219\u3002\n- \u4F7F\u7528\u672C\u63D2\u4EF6\u9020\u6210\u7684\u4EFB\u4F55\u540E\u679C\u5C06\u7531\u60A8\u4E2A\u4EBA\u627F\u62C5\uFF0C**\u5305\u62EC\u4F46\u4E0D\u9650\u4E8E\u8D26\u53F7\u88AB\u9650\u5236\u3001\u5C01\u7981\u7B49**\u3002\n\n\u53D7\u5E73\u53F0\u670D\u52A1\u7AEF\u9650\u5236\uFF0C\u672C\u63D2\u4EF6\u751F\u6210\u7684\u5408\u5E76\u8F6C\u53D1\u5E76\u975E\u5B8C\u7F8E\uFF0C\u5176\u7F3A\u9677\u53EF\u4EE5\u88AB\u4EBA\u8F7B\u6613\u8BC6\u7834\uFF0C\u5E73\u53F0\u4E5F\u80FD\u68C0\u6D4B\u51FA\u5F02\u5E38\uFF0C\u8BF7\u52FF\u7528\u4E8E\u6B3A\u9A97\u3001\u5192\u5145\u3001\u9A9A\u6270\u6216\u5176\u4ED6\u8FDD\u6CD5\u8FDD\u89C4\u7528\u9014\u3002\n\n---\n\n**\u4F7F\u7528\u524D\u8BF7\u786E\u8BA4\u5DF2\u5F00\u542F\u4EE5\u4E0B\u9009\u9879\uFF0C\u5426\u5219\u672C\u63D2\u4EF6\u53EF\u80FD\u65E0\u6CD5\u83B7\u53D6\u5230\u5408\u5E76\u8F6C\u53D1\u6240\u9700\u7684\u4E0A\u4E0B\u6587\u4FE1\u606F\uFF1A**\n- \u5728 chatluna \u63D2\u4EF6\u7684\u201C\u5BF9\u8BDD\u884C\u4E3A\u9009\u9879\u201D\u4E2D\u542F\u7528\uFF1AattachForwardMsgIdToContext\u3002\n- \u5728 chatluna-character \u63D2\u4EF6\u7684\u201C\u5BF9\u8BDD\u8BBE\u7F6E\u201D\u4E2D\u542F\u7528\uFF1AenableMessageId\u3002\n";
export {};
