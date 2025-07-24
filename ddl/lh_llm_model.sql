CREATE TABLE `lh_llm_model` (
    `id` int NOT NULL AUTO_INCREMENT,
    `llm` varchar(255) NOT NULL,
    `user_id` int NOT NULL,
    `app_key` varchar(255) NOT NULL,
    `status` tinyint DEFAULT '1' COMMENT '状态：1-启用，0-禁用',
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    KEY `idx_llm` (`llm`),
    KEY `idx_user_id` (`user_id`),
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

