CREATE TABLE `bookmarks` (
	`lesson_id` text PRIMARY KEY NOT NULL,
	`note` text DEFAULT '' NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `card_state` (
	`card_id` text PRIMARY KEY NOT NULL,
	`ease` real DEFAULT 2.5 NOT NULL,
	`interval_days` real DEFAULT 0 NOT NULL,
	`reps` integer DEFAULT 0 NOT NULL,
	`lapses` integer DEFAULT 0 NOT NULL,
	`due_at` integer NOT NULL,
	`last_reviewed_at` integer
);
--> statement-breakpoint
CREATE TABLE `concept_edges` (
	`concept_id` text NOT NULL,
	`prerequisite_id` text NOT NULL,
	PRIMARY KEY(`concept_id`, `prerequisite_id`),
	FOREIGN KEY (`concept_id`) REFERENCES `concepts`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`prerequisite_id`) REFERENCES `concepts`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `concept_mastery` (
	`concept_id` text PRIMARY KEY NOT NULL,
	`score` real NOT NULL,
	`attempts` integer DEFAULT 0 NOT NULL,
	`recent_failures` integer DEFAULT 0 NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `concepts` (
	`id` text PRIMARY KEY NOT NULL,
	`title` text NOT NULL,
	`description` text NOT NULL,
	`stage` text NOT NULL
);
--> statement-breakpoint
CREATE TABLE `courses` (
	`id` text PRIMARY KEY NOT NULL,
	`title` text NOT NULL,
	`summary` text NOT NULL,
	`stage` text NOT NULL,
	`kind` text NOT NULL,
	`position` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `evidence` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`capstone_id` text NOT NULL,
	`criterion` text NOT NULL,
	`text` text NOT NULL,
	`self_rating` integer NOT NULL,
	`at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `exercise_attempts` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`exercise_id` text NOT NULL,
	`passed` integer NOT NULL,
	`runtime` text NOT NULL,
	`output` text NOT NULL,
	`at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `exercise_state` (
	`exercise_id` text PRIMARY KEY NOT NULL,
	`code` text NOT NULL,
	`solved` integer DEFAULT false NOT NULL,
	`hints_used` integer DEFAULT 0 NOT NULL,
	`solution_revealed` integer DEFAULT false NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `exercises` (
	`id` text PRIMARY KEY NOT NULL,
	`lesson_id` text NOT NULL,
	`title` text NOT NULL,
	`difficulty` integer NOT NULL,
	`runtime` text NOT NULL,
	`prompt` text NOT NULL,
	`starter` text NOT NULL,
	`tests` text NOT NULL,
	`hints` text NOT NULL,
	`solution` text NOT NULL,
	`concept_ids` text NOT NULL,
	FOREIGN KEY (`lesson_id`) REFERENCES `lessons`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `flashcards` (
	`id` text PRIMARY KEY NOT NULL,
	`lesson_id` text,
	`concept_id` text NOT NULL,
	`front` text NOT NULL,
	`back` text NOT NULL,
	FOREIGN KEY (`lesson_id`) REFERENCES `lessons`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`concept_id`) REFERENCES `concepts`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `lesson_concepts` (
	`lesson_id` text NOT NULL,
	`concept_id` text NOT NULL,
	`role` text NOT NULL,
	PRIMARY KEY(`lesson_id`, `concept_id`, `role`),
	FOREIGN KEY (`lesson_id`) REFERENCES `lessons`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`concept_id`) REFERENCES `concepts`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `lesson_progress` (
	`lesson_id` text PRIMARY KEY NOT NULL,
	`status` text NOT NULL,
	`last_visited_at` integer NOT NULL,
	`completed_at` integer
);
--> statement-breakpoint
CREATE TABLE `lessons` (
	`id` text PRIMARY KEY NOT NULL,
	`course_id` text NOT NULL,
	`position` integer NOT NULL,
	`title` text NOT NULL,
	`summary` text DEFAULT '' NOT NULL,
	`kind` text NOT NULL,
	`minutes` integer DEFAULT 30 NOT NULL,
	`body` text NOT NULL,
	`sources` text NOT NULL,
	`hash` text NOT NULL,
	FOREIGN KEY (`course_id`) REFERENCES `courses`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `lessons_course_idx` ON `lessons` (`course_id`,`position`);--> statement-breakpoint
CREATE TABLE `notebooks` (
	`id` text PRIMARY KEY NOT NULL,
	`title` text NOT NULL,
	`cells` text NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `questions` (
	`id` text PRIMARY KEY NOT NULL,
	`lesson_id` text NOT NULL,
	`concept_id` text NOT NULL,
	`kind` text NOT NULL,
	`stage` text DEFAULT 'check' NOT NULL,
	`prompt` text NOT NULL,
	`options` text NOT NULL,
	`answer` text NOT NULL,
	`explanation` text NOT NULL,
	FOREIGN KEY (`lesson_id`) REFERENCES `lessons`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`concept_id`) REFERENCES `concepts`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `quiz_attempts` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`question_id` text NOT NULL,
	`concept_id` text NOT NULL,
	`correct` integer NOT NULL,
	`response` text NOT NULL,
	`at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `quiz_attempts_q` ON `quiz_attempts` (`question_id`);--> statement-breakpoint
CREATE TABLE `review_log` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`card_id` text NOT NULL,
	`concept_id` text NOT NULL,
	`grade` integer NOT NULL,
	`at` integer NOT NULL
);
